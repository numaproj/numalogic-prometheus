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
from numaprom.tools import get_metric_config, save_model

LOGGER = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame, limit=12) -> pd.DataFrame:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.fillna(method="ffill", limit=limit)
    df = df.fillna(method="bfill", limit=limit)
    if df.columns[df.isna().any()].tolist():
        df.dropna(inplace=True)
    return df


def _fetch_data(metric_name: str, model_config: dict, labels: dict) -> pd.DataFrame:
    _start_time = time.time()

    prometheus_server = os.getenv("PROMETHEUS_SERVER", DEFAULT_PROMETHEUS_SERVER)
    datafetcher = Prometheus(prometheus_server)

    end_dt = datetime.now(pytz.utc)
    start_dt = end_dt - timedelta(hours=15)

    df = datafetcher.query_metric(
        metric_name=metric_name,
        labels_map=labels,
        start=start_dt.timestamp(),
        end=end_dt.timestamp(),
        step=model_config["scrape_interval"]
    )
    LOGGER.info("Time taken to fetch data: %s, for df shape: %s", time.time() - _start_time, df.shape)
    return df


def _train_model(x, model_config):
    _start_train = time.time()

    win_size = model_config["win_size"]
    dataset = StreamingDataset(x, win_size)
    model = SparseVanillaAE(seq_len=win_size)

    trainer = AutoencoderTrainer(max_epochs=40)
    trainer.fit(model, train_dataloaders=DataLoader(dataset, batch_size=64))

    LOGGER.info("Time taken to train model: %s", time.time() - _start_train)
    return model


def _preprocess(x_raw):
    clf = LogTransformer()
    x_scaled = clf.fit_transform(x_raw)
    return x_scaled


def train(datums: List[Datum]) -> Responses:
    responses = Responses()

    for _datum in datums:
        payload = orjson.loads(_datum.value)

        namespace = payload["namespace"]
        metric_name = payload["name"]

        metric_config = get_metric_config(metric_name)
        model_config = metric_config["model_config"]
        win_size = model_config["win_size"]

        train_df = _fetch_data(metric_name, model_config, {"namespace": namespace})
        train_df = clean_data(train_df)

        if len(train_df) < model_config["win_size"]:
            warn_msg = f"Skipping training since traindata size: {train_df.shape} " \
                       f"is less than winsize: {win_size}"
            LOGGER.warning(warn_msg)
            responses.append(Response.as_failure(_datum.id, err_msg=warn_msg))
            continue

        x_train = _preprocess(train_df.to_numpy())
        model = _train_model(x_train, model_config)

        skeys = [namespace, metric_name]
        print("SAVING NOW!!!", save_model)
        version = save_model(
            skeys=skeys,
            dkeys=[model_config["model_name"]],
            model=model
        )
        LOGGER.info("Model saved with skeys: %s with version: %s", skeys, version)
        responses.append(Response.as_success(_datum.id))

    return responses
