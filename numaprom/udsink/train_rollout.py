import os
import time
from collections.abc import Iterator

import numpy as np
from numalogic.tools.exceptions import InvalidDataShapeError
import pandas as pd
from numalogic.config import PreprocessFactory, ModelInfo, ThresholdFactory, ModelFactory
from numalogic.models.autoencoder import AutoencoderTrainer
from numalogic.registry import RedisRegistry
from numalogic.tools.data import StreamingDataset
from numalogic.tools.exceptions import RedisRegistryError
from numalogic.tools.types import redis_client_t
from orjson import orjson
from pynumaflow.sinker import Datum, Responses, Response
from sklearn.pipeline import make_pipeline
from torch.utils.data import DataLoader

from numaprom import LOGGER, MetricConf
from numaprom.clients.sentinel import get_redis_client_from_conf
from numaprom.entities import TrainerPayload
from numaprom.tools import fetch_data
from numaprom.watcher import ConfigManager

REQUEST_EXPIRY = int(os.getenv("REQUEST_EXPIRY", "300"))
# REDIS_CLIENT = get_redis_client_from_conf(master_node=True, recreate=True)
REDIS_CLIENT_MASTER = get_redis_client_from_conf(master_node=True, reset=True)


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
    df = df.reset_index(drop=True)
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

    LOGGER.debug(
        "{uuid} - Time taken to train model: {time} sec",
        uuid=uuid,
        time=time.perf_counter() - _start_train,
    )

    train_reconerr = trainer.predict(model, dataloaders=DataLoader(dataset, batch_size=64))
    # return the trainer to avoid Weakreference error
    return train_reconerr.numpy(), model, trainer


def _preprocess(x_raw, preproc_cfgs: list[ModelInfo]):
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


def _is_new_request(redis_client: redis_client_t, payload: TrainerPayload) -> bool:
    _ckeys = ":".join([payload.composite_keys["namespace"], payload.composite_keys["name"]])
    r_key = f"trainrollout::{_ckeys}"
    value = redis_client.get(r_key)
    if value:
        return False

    redis_client.setex(r_key, time=REQUEST_EXPIRY, value=1)
    return True


def get_model_config(metric_config):
    model_cfg = metric_config.numalogic_conf.model
    model_factory = ModelFactory()
    model = model_factory.get_instance(model_cfg)
    return model


def train_rollout(datums: Iterator[Datum]) -> Responses:
    global REDIS_CLIENT_MASTER
    redis_client = REDIS_CLIENT_MASTER
    responses = Responses()

    for _datum in datums:
        payload = TrainerPayload(**orjson.loads(_datum.value))

        LOGGER.debug(
            "{uuid} - Starting Training for keys: {skeys}",
            uuid=payload.uuid,
            skeys=payload.composite_keys,
        )

        is_new = _is_new_request(redis_client, payload)
        if not is_new:
            LOGGER.debug(
                "{uuid} - Skipping rollouts train request with keys: {skeys}",
                uuid=payload.uuid,
                skeys=payload.composite_keys,
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
            LOGGER.error(
                "{uuid} - KeyError while data cleaning for train payload: {payload}",
                uuid=payload.uuid,
                payload=payload,
            )
            responses.append(Response.as_success(_datum.id))
            continue

        if train_df.shape[1] != get_model_config(metric_config).n_features:
            LOGGER.error(
                "Expected Shape is: {shape}", shape=get_model_config(metric_config).n_features
            )
            raise InvalidDataShapeError(f"Train data shape error. Input shape: {train_df.shape}")

        if len(train_df) < metric_config.min_train_size:
            LOGGER.warning(
                "{uuid} - Skipping training, train data less than minimum "
                "required: {min_train_size}, df shape: {shape}",
                uuid=payload.uuid,
                min_train_size=metric_config.min_train_size,
                shape=train_df.shape,
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

    # TODO if one of the models fail to save, delete the previously saved models & transition stage
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
        LOGGER.exception(
            "{uuid} - Error while saving Model with skeys: {skeys}, err: {err}",
            uuid=payload.uuid,
            skeys=skeys,
            err=err,
        )
    else:
        LOGGER.info(
            "{uuid} - Model saved with skeys: {skeys} with version: {version}",
            uuid=payload.uuid,
            skeys=skeys,
            version=version,
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
        LOGGER.exception(
            "{uuid} - Error while saving Preproc model with skeys: {skeys}, err: {err}",
            uuid=payload.uuid,
            skeys=skeys,
            err=err,
        )
    else:
        LOGGER.info(
            "{uuid} - Preproc model saved with skeys: {skeys} with version: {version}",
            uuid=payload.uuid,
            skeys=skeys,
            version=version,
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
        LOGGER.error(
            "{uuid} - Error while saving Threshold model with skeys: {skeys}, err: {err}",
            uuid=payload.uuid,
            skeys=skeys,
            err=err,
        )
    else:
        LOGGER.info(
            "{uuid} - Threshold model saved with skeys: {skeys} with version: {version}",
            uuid=payload.uuid,
            skeys=skeys,
            version=version,
        )
