import logging
import os
import time
from typing import List

import numpy as np
import pandas as pd
from numalogic.models.autoencoder import AutoencoderTrainer
from numalogic.models.autoencoder.variants import SparseVanillaAE
from numalogic.tools.data import StreamingDataset
from orjson import orjson
from pynumaflow.sink import Datum, Responses, Response
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from numaprom.redis import get_redis_client
from numaprom.tools import get_metric_config, save_model, fetch_data

_LOGGER = logging.getLogger(__name__)

HOST = os.getenv("REDIS_HOST")
PORT = os.getenv("REDIS_PORT")
AUTH = os.getenv("REDIS_AUTH")
EXPIRY = int(os.getenv("REDIS_EXPIRY", 300))


def clean_data(df: pd.DataFrame, limit=12) -> pd.DataFrame:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.fillna(method="ffill", limit=limit)
    df = df.fillna(method="bfill", limit=limit)
    if df.columns[df.isna().any()].tolist():
        df.dropna(inplace=True)
    return df


def _train_model(uuid, x, model_config):
    _start_train = time.time()

    win_size = model_config["win_size"]
    dataset = StreamingDataset(x, win_size)
    model = SparseVanillaAE(seq_len=win_size)

    trainer = AutoencoderTrainer(max_epochs=40)
    trainer.fit(model, train_dataloaders=DataLoader(dataset, batch_size=64))

    _LOGGER.debug("%s - Time taken to train model: %s", uuid, time.perf_counter() - _start_train)
    return model


def _preprocess(x_raw):
    clf = StandardScaler()
    x_scaled = clf.fit_transform(x_raw)
    return x_scaled, clf


def _is_new_request(namespace: str, metric: str) -> bool:
    redis_client = get_redis_client(HOST, PORT, password=AUTH, recreate=False)
    r_key = f"train::{namespace}:{metric}"

    value = redis_client.get(r_key)
    if value:
        return False

    redis_client.setex(r_key, time=EXPIRY, value=1)
    return True


def train(datums: List[Datum]) -> Responses:
    responses = Responses()

    for _datum in datums:
        payload = orjson.loads(_datum.value)

        _id = payload["uuid"]
        namespace = payload["namespace"]
        metric_name = payload["name"]

        _LOGGER.debug(
            "%s - Starting Training for namespace: %s, metric: %s", _id, namespace, metric_name
        )

        is_new = _is_new_request(namespace, metric_name)
        if not is_new:
            _LOGGER.debug(
                "%s - Skipping train request with namespace: %s, metric: %s",
                _id,
                namespace,
                metric_name,
            )
            responses.append(Response.as_success(_datum.id))
            continue

        metric_config = get_metric_config(metric_name)
        model_config = metric_config["model_config"]
        win_size = model_config["win_size"]

        train_df = fetch_data(_id, metric_name, model_config, {"namespace": namespace})
        train_df = clean_data(train_df)

        if len(train_df) < model_config["win_size"]:
            _LOGGER.info(
                "%s - Skipping training since traindata size: %s is less than winsize: %s",
                _id,
                train_df.shape,
                win_size,
            )
            responses.append(Response.as_success(_datum.id))
            continue

        x_train, preproc_clf = _preprocess(train_df.to_numpy())
        model = _train_model(_id, x_train, model_config)

        skeys = [namespace, metric_name]

        version = save_model(
            skeys=skeys, dkeys=["preproc"], model=preproc_clf, artifact_type="sklearn"
        )
        _LOGGER.info(
            "%s - Preproc model saved with skeys: %s with version: %s", _id, skeys, version
        )

        version = save_model(skeys=skeys, dkeys=[model_config["model_name"]], model=model)
        _LOGGER.info("%s - Model saved with skeys: %s with version: %s", _id, skeys, version)

        responses.append(Response.as_success(_datum.id))

    return responses
