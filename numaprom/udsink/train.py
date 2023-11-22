import os
import time

import numpy as np
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

from numaprom import LOGGER
from numaprom.clients.sentinel import get_redis_client_from_conf
from numaprom.entities import TrainerPayload
from numaprom.tools import fetch_data
from numaprom.watcher import ConfigManager


REQUEST_EXPIRY = int(os.getenv("REQUEST_EXPIRY", 300))


def clean_data(df: pd.DataFrame, limit=12) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.fillna(method="ffill", limit=limit)
    df = df.fillna(method="bfill", limit=limit)
    if df.columns[df.isna().any()].tolist():
        df.dropna(inplace=True)

    df.set_index("timestamp", inplace=True)
    return df


def _train_model(uuid, x, model_cfg, trainer_cfg):
    _start_train = time.time()

    model_factory = ModelFactory()
    model = model_factory.get_instance(model_cfg)
    dataset = StreamingDataset(x, model.seq_len)

    trainer = AutoencoderTrainer(**trainer_cfg)
    trainer.fit(model, train_dataloaders=DataLoader(dataset, batch_size=64))

    LOGGER.debug(
        "{uuid} - Time taken to train model: {time}",
        uuid=uuid,
        time=time.perf_counter() - _start_train,
    )

    train_reconerr = trainer.predict(model, dataloaders=DataLoader(dataset, batch_size=64))
    return train_reconerr.numpy(), model


def _preprocess(x_raw, preproc_cfg: list[ModelInfo]):
    preproc_factory = PreprocessFactory()

    if len(preproc_cfg) > 1:
        preproc_clfs = []
        for _cfg in preproc_cfg:
            _clf = preproc_factory.get_instance(_cfg)
            preproc_clfs.append(_clf)
        preproc_clf = make_pipeline(*preproc_clfs)
    else:
        preproc_clf = preproc_factory.get_instance(preproc_cfg[0])

    x_scaled = preproc_clf.fit_transform(x_raw)
    return x_scaled, preproc_clf


def _find_threshold(x_reconerr, thresh_cfg: ModelInfo):
    thresh_factory = ThresholdFactory()
    thresh_clf = thresh_factory.get_instance(thresh_cfg)
    thresh_clf.fit(x_reconerr)
    return thresh_clf


def _is_new_request(redis_client: redis_client_t, payload: TrainerPayload) -> bool:
    _ckeys = ":".join([payload.composite_keys["namespace"], payload.composite_keys["name"]])
    r_key = f"train::{_ckeys}"

    value = redis_client.get(r_key)
    if value:
        return False

    redis_client.setex(r_key, time=REQUEST_EXPIRY, value=1)
    return True


def train(datums: list[Datum]) -> Responses:
    responses = Responses()
    redis_client = get_redis_client_from_conf()

    for _datum in datums:
        payload = TrainerPayload(**orjson.loads(_datum.value))

        LOGGER.debug(
            "{uuid} - Starting Training for keys: {keys}",
            uuid=payload.uuid,
            keys=payload.composite_keys,
        )

        is_new = _is_new_request(redis_client, payload)
        if not is_new:
            LOGGER.debug(
                "{uuid} - Skipping train request with keys: {keys}",
                uuid=payload.uuid,
                keys=payload.composite_keys,
            )
            responses.append(Response.as_success(_datum.id))
            continue

        metric_config = ConfigManager.get_metric_config(payload.composite_keys)
        model_cfg = metric_config.numalogic_conf.model

        train_df = fetch_data(
            payload,
            metric_config,
            {"namespace": payload.composite_keys["namespace"]},
            hours=metric_config.train_hours,
        )
        train_df = clean_data(train_df)

        if len(train_df) < metric_config.min_train_size:
            LOGGER.warning(
                "{uuid} - Skipping training, train data less than "
                "minimum required: {min_train_size}, df shape: {shape}",
                uuid=payload.uuid,
                min_train_size=metric_config.min_train_size,
                shape=train_df.shape,
            )
            responses.append(Response.as_success(_datum.id))
            continue

        preproc_cfgs = metric_config.numalogic_conf.preprocess
        x_train, preproc_clf = _preprocess(train_df.to_numpy(), preproc_cfgs)

        trainer_cfg = metric_config.numalogic_conf.trainer
        x_reconerr, anomaly_model = _train_model(payload.uuid, x_train, model_cfg, trainer_cfg)

        thresh_cfg = metric_config.numalogic_conf.threshold
        thresh_clf = _find_threshold(x_reconerr, thresh_cfg)

        # TODO change this to just use **composite_keys
        skeys = [payload.composite_keys["namespace"], payload.composite_keys["name"]]

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

        responses.append(Response.as_success(_datum.id))

    return responses
