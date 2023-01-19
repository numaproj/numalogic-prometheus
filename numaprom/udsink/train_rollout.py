import logging
import os
import time
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd
import pytz
from numalogic.models.autoencoder import AutoencoderTrainer
from numalogic.models.autoencoder.variants import SparseVanillaAE
from numalogic.preprocess.transformer import LogTransformer
from numalogic.tools.data import StreamingDataset
from orjson import orjson
from pynumaflow.sink import Datum, Responses, Response
from torch.utils.data import DataLoader

from numaprom._constants import DEFAULT_PROMETHEUS_SERVER
from numaprom.prometheus import Prometheus
from numaprom.redis import get_redis_client
from numaprom.tools import get_metric_config, save_model

LOGGER = logging.getLogger(__name__)

HOST = os.getenv("REDIS_HOST")
PORT = os.getenv("REDIS_PORT")
AUTH = os.getenv("REDIS_AUTH")
EXPIRY = int(os.getenv("REDIS_EXPIRY", 360))


# TODO: extract all good hashes, including when there are 2 hashes at a time
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
    if len(df) < (1.5 * 60 * 12):
        LOGGER.exception("Not enough training points to initiate training")
        return pd.DataFrame()
    return df


def _fetch_data(metric_name: str, model_config: dict, labels: dict, return_labels=None) -> pd.DataFrame:
    _start_time = time.time()

    prometheus_server = os.getenv("PROMETHEUS_SERVER", DEFAULT_PROMETHEUS_SERVER)
    datafetcher = Prometheus(prometheus_server)

    end_dt = datetime.now(pytz.utc)
    start_dt = end_dt - timedelta(hours=15)

    df = datafetcher.query_metric(
        metric_name=metric_name,
        labels_map=labels,
        return_labels=return_labels,
        start=start_dt.timestamp(),
        end=end_dt.timestamp(),
        step=model_config["scrape_interval"],
    )
    LOGGER.info(
        "Time taken to fetch data: %s, for df shape: %s", time.time() - _start_time, df.shape
    )
    return df


def _train_model(x, model_config):
    _start_train = time.time()

    win_size = model_config["win_size"]
    dataset = StreamingDataset(x, win_size)
    model = SparseVanillaAE(seq_len=win_size)

    trainer = AutoencoderTrainer(max_epochs=40)
    trainer.fit(model, train_dataloaders=DataLoader(dataset, batch_size=64))

    LOGGER.debug("Time taken to train model: %s", time.time() - _start_train)
    return model


def _preprocess(x_raw):
    clf = LogTransformer()
    x_scaled = clf.fit_transform(x_raw)
    return x_scaled


def _is_new_request(namespace: str, metric: str) -> bool:
    redis_client = get_redis_client(HOST, PORT, password=AUTH, recreate=False)
    r_key = f"trainrollout::{namespace}:{metric}"

    value = redis_client.get(r_key)
    if value:
        return False

    redis_client.setex(r_key, time=EXPIRY, value=1)
    return True


def train_rollout(datums: List[Datum]) -> Responses:
    responses = Responses()

    for _datum in datums:
        payload = orjson.loads(_datum.value)

        _id = payload.get("uuid")
        namespace = payload["namespace"]
        metric_name = payload["name"]

        is_new = _is_new_request(namespace, metric_name)
        if not is_new:
            LOGGER.info(
                "%s - Skipping rollouts train request with namespace: %s, metric: %s",
                _id,
                namespace,
                metric_name,
            )
            responses.append(Response.as_success(_datum.id))
            continue

        metric_config = get_metric_config(metric_name)
        model_config = metric_config["model_config"]
        win_size = model_config["win_size"]

        train_df = _fetch_data(metric_name, model_config, {"namespace": namespace},
                               return_labels=["hash_id"])
        train_df = clean_data(train_df, "hash_id")

        if len(train_df) < model_config["win_size"]:
            _info_msg = (
                f"{_id} - Skipping training since traindata size: {train_df.shape} "
                f"is less than winsize: {win_size}"
            )
            LOGGER.info(_info_msg)
            responses.append(Response.as_failure(_datum.id, err_msg=_info_msg))
            continue

        x_train = _preprocess(train_df.to_numpy())
        model = _train_model(x_train, model_config)

        skeys = [namespace, metric_name]
        version = save_model(skeys=skeys, dkeys=[model_config["model_name"]], model=model)
        LOGGER.info("%s - Model saved with skeys: %s with version: %s", _id, skeys, version)
        responses.append(Response.as_success(_datum.id))

    return responses
