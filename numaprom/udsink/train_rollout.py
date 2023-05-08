import os
import time
from typing import List, Iterator

import numpy as np
import pandas as pd
from numalogic.config import PreprocessFactory, ModelInfo, ThresholdFactory, ModelFactory
from numalogic.models.autoencoder import AutoencoderTrainer
from numalogic.registry import RedisRegistry
from numalogic.tools.data import StreamingDataset
from numalogic.tools.exceptions import RedisRegistryError
from numalogic.tools.types import redis_client_t
from orjson import orjson
from pynumaflow.sink import Datum, Responses, Response
from sklearn.pipeline import make_pipeline
from torch.utils.data import DataLoader

from numaprom import get_logger, MetricConf
from numaprom.clients.sentinel import get_redis_client
from numaprom.entities import TrainerPayload
from numaprom.tools import fetch_data
from numaprom.watcher import ConfigManager

_LOGGER = get_logger(__name__)

AUTH = os.getenv("REDIS_AUTH")


# TODO: extract all good hashes, including when there are 2 hashes at a time
# TODO avoid filling inf with nan, or at least throw warning
def clean_data(df: pd.DataFrame, hash_col: str, limit=12) -> pd.DataFrame:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.fillna(method="ffill", limit=limit)
    df = df.fillna(method="bfill", limit=limit)
    if df.columns[df.isna().any()].tolist():
        df.dropna(inplace=True)

    if df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    df = (
        pd.merge(df, df[df.duplicated("timestamp", keep=False)], indicator=True, how="outer")
        .query('_merge=="left_only"')
        .drop("_merge", axis=1)
    )
    df.set_index("timestamp", inplace=True)
    df.drop(hash_col, axis=1, inplace=True)
    df = df.sort_values(by=["timestamp"], ascending=True)
    return df


def _train_model(uuid, x, model_cfg, trainer_cfg):
    _start_train = time.perf_counter()

    model_factory = ModelFactory()
    model = model_factory.get_instance(model_cfg)
    dataset = StreamingDataset(x, model.seq_len)

    trainer = AutoencoderTrainer(**trainer_cfg)
    trainer.fit(model, train_dataloaders=DataLoader(dataset, batch_size=64))

    _LOGGER.debug(
        "%s - Time taken to train model: %.3f sec", uuid, time.perf_counter() - _start_train
    )

    train_reconerr = trainer.predict(model, dataloaders=DataLoader(dataset, batch_size=64))
    # return the trainer to avoid Weakreference error
    return train_reconerr.numpy(), model, trainer


def _preprocess(x_raw, preproc_cfgs: List[ModelInfo]):
    preproc_factory = PreprocessFactory()
    preproc_clfs = []
    for _cfg in preproc_cfgs:
        _clf = preproc_factory.get_instance(_cfg)
        preproc_clfs.append(_clf)
    preproc_pl = make_pipeline(*preproc_clfs)

    x_scaled = preproc_pl.fit_transform(x_raw)
    return x_scaled, preproc_pl


def _find_threshold(x_reconerr, thresh_cfg: ModelInfo):
    thresh_factory = ThresholdFactory()
    thresh_clf = thresh_factory.get_instance(thresh_cfg)
    thresh_clf.fit(x_reconerr)
    return thresh_clf


def _is_new_request(redis_client, redis_conf, payload: TrainerPayload) -> bool:
    _ckeys = ":".join([payload.composite_keys["namespace"], payload.composite_keys["name"]])
    r_key = f"trainrollout::{_ckeys}"
    value = redis_client.get(r_key)
    if value:
        return False

    redis_client.setex(r_key, time=redis_conf.expiry, value=1)
    return True


def train_rollout(datums: Iterator[Datum]) -> Responses:
    responses = Responses()
    redis_conf = ConfigManager.get_redis_config()
    redis_client = get_redis_client(
        redis_conf.host,
        redis_conf.port,
        password=AUTH,
        mastername=redis_conf.master_name,
        recreate=False,
    )

    for _datum in datums:
        payload = TrainerPayload(**orjson.loads(_datum.value))

        _LOGGER.debug("%s - Starting Training for keys: %s", payload.uuid, payload.composite_keys)

        is_new = _is_new_request(redis_client, redis_conf, payload)
        if not is_new:
            _LOGGER.debug(
                "%s - Skipping rollouts train request with keys: %s",
                payload.uuid,
                payload.composite_keys,
            )
            responses.append(Response.as_success(_datum.id))
            continue

        metric_config = ConfigManager.get_metric_config(payload.composite_keys)

        # TODO: standardize the label name
        if "rollouts_pod_template_hash" in payload.composite_keys:
            hash_label = "rollouts_pod_template_hash"
        else:
            hash_label = "hash_id"

        train_df = fetch_data(
            payload,
            metric_config,
            {"namespace": payload.composite_keys["namespace"]},
            return_labels=[hash_label],
            hours=metric_config.train_hours,
        )
        try:
            train_df = clean_data(train_df, hash_label)
        except KeyError:
            _LOGGER.error(
                "%s - KeyError while data cleaning for train payload: %s", payload.uuid, payload
            )
            responses.append(Response.as_success(_datum.id))
            continue

        if len(train_df) < metric_config.min_train_size:
            _LOGGER.warning(
                "%s - Skipping training, train data less than minimum required: %s, df shape: %s",
                payload.uuid,
                metric_config.min_train_size,
                train_df.shape,
            )
            responses.append(Response.as_success(_datum.id))
            continue

        _train_and_save(metric_config, payload, redis_client, train_df)

        responses.append(Response.as_success(_datum.id))

    return responses


def _train_and_save(
    metric_config: MetricConf,
    payload: TrainerPayload,
    redis_client: redis_client_t,
    train_df: pd.DataFrame,
) -> None:
    model_cfg = metric_config.numalogic_conf.model
    preproc_cfgs = metric_config.numalogic_conf.preprocess

    x_train, preproc_clf = _preprocess(train_df.to_numpy(), preproc_cfgs)

    trainer_cfg = metric_config.numalogic_conf.trainer
    x_reconerr, anomaly_model, trainer = _train_model(payload.uuid, x_train, model_cfg, trainer_cfg)

    thresh_cfg = metric_config.numalogic_conf.threshold
    thresh_clf = _find_threshold(x_reconerr, thresh_cfg)

    skeys = [payload.composite_keys["namespace"], payload.composite_keys["name"]]

    # TODO if one of the models fail to save, delete the previously saved models and transition stage
    # Save main model
    model_registry = RedisRegistry(client=redis_client)
    try:
        version = model_registry.save(
            skeys=skeys,
            dkeys=[model_cfg.name],
            artifact=anomaly_model,
            uuid=payload.uuid,
            train_size=train_df.shape[0],
        )
    except RedisRegistryError as err:
        _LOGGER.exception(
            "%s - Error while saving Model with skeys: %s, err: %r", payload.uuid, skeys, err
        )
    else:
        _LOGGER.info(
            "%s - Model saved with skeys: %s with version: %s", payload.uuid, skeys, version
        )
    # Save preproc model
    try:
        version = model_registry.save(
            skeys=skeys,
            dkeys=[_conf.name for _conf in preproc_cfgs],
            artifact=preproc_clf,
            uuid=payload.uuid,
        )
    except RedisRegistryError as err:
        _LOGGER.exception(
            "%s - Error while saving Preproc model with skeys: %s, err: %r",
            payload.uuid,
            skeys,
            err,
        )
    else:
        _LOGGER.info(
            "%s - Preproc model saved with skeys: %s with version: %s",
            payload.uuid,
            skeys,
            version,
        )
    # Save threshold model
    try:
        version = model_registry.save(
            skeys=skeys,
            dkeys=[thresh_cfg.name],
            artifact=thresh_clf,
            uuid=payload.uuid,
        )
    except RedisRegistryError as err:
        _LOGGER.error(
            "%s - Error while saving Threshold model with skeys: %s, err: %r",
            payload.uuid,
            skeys,
            err,
        )
    else:
        _LOGGER.info(
            "%s - Threshold model saved with skeys: %s with version: %s",
            payload.uuid,
            skeys,
            version,
        )
