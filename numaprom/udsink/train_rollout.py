import logging
import os
import time
from typing import List

import numpy as np
import pandas as pd
from numalogic.models.autoencoder import AutoencoderTrainer
from numalogic.models.autoencoder.variants import SparseVanillaAE
from numalogic.models.threshold import StdDevThreshold
from numalogic.tools.data import StreamingDataset
from orjson import orjson
from pynumaflow.sink import Datum, Responses, Response
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from numaprom.entities import TrainerPayload
from numaprom.redis import get_redis_client
from numaprom.tools import get_metric_config, save_model, fetch_data

_LOGGER = logging.getLogger(__name__)

HOST = os.getenv("REDIS_HOST")
PORT = os.getenv("REDIS_PORT")
AUTH = os.getenv("REDIS_AUTH")
EXPIRY = int(os.getenv("REDIS_EXPIRY", 360))
MIN_TRAIN_SIZE = int(os.getenv("MIN_TRAIN_SIZE", 1000))


# TODO: extract all good hashes, including when there are 2 hashes at a time
# TODO avoid filling inf with nan, or at least throw warning
def clean_data(uuid: str, df: pd.DataFrame, hash_col: str, limit=12) -> pd.DataFrame:
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
    if len(df) < MIN_TRAIN_SIZE:
        _LOGGER.error(
            "%s - Train data less than minimum required: %s, df shape: %s",
            uuid,
            MIN_TRAIN_SIZE,
            df.shape,
        )
        return pd.DataFrame()
    return df


def _train_model(uuid, x, model_config):
    _start_train = time.perf_counter()

    win_size = model_config["win_size"]
    dataset = StreamingDataset(x, win_size)
    model = SparseVanillaAE(seq_len=win_size)

    trainer = AutoencoderTrainer(max_epochs=40)
    trainer.fit(model, train_dataloaders=DataLoader(dataset, batch_size=64))

    _LOGGER.debug(
        "%s - Time taken to train model: %.3f sec", uuid, time.perf_counter() - _start_train
    )

    train_reconerr = trainer.predict(model, dataloaders=DataLoader(dataset, batch_size=64))
    return train_reconerr.numpy(), model


def _preprocess(x_raw):
    clf = StandardScaler()
    x_scaled = clf.fit_transform(x_raw)
    return x_scaled, clf


def _find_threshold(x_reconerr):
    clf = StdDevThreshold()
    clf.fit(x_reconerr)
    return clf


def _is_new_request(payload: TrainerPayload) -> bool:
    redis_client = get_redis_client(HOST, PORT, password=AUTH, recreate=False)
    _ckeys = ":".join([payload.composite_keys["namespace"], payload.composite_keys["name"]])
    r_key = f"trainrollout::{_ckeys}"
    value = redis_client.get(r_key)
    if value:
        return False

    redis_client.setex(r_key, time=EXPIRY, value=1)
    return True


def train_rollout(datums: List[Datum]) -> Responses:
    responses = Responses()

    for _datum in datums:
        payload = TrainerPayload(**orjson.loads(_datum.value))

        _LOGGER.debug("%s - Starting Training for keys: %s", payload.uuid, payload.composite_keys)

        is_new = _is_new_request(payload)
        if not is_new:
            _LOGGER.debug(
                "%s - Skipping rollouts train request with keys: %s",
                payload.uuid,
                payload.composite_keys,
            )
            responses.append(Response.as_success(_datum.id))
            continue

        metric_config = get_metric_config(payload.composite_keys["name"])
        model_config = metric_config["model_config"]
        win_size = model_config["win_size"]

        train_df = fetch_data(
            payload,
            metric_config,
            {"namespace": payload.composite_keys["namespace"]},
            return_labels=["hash_id"],
        )
        try:
            train_df = clean_data(payload.uuid, train_df, "hash_id")
        except KeyError:
            _LOGGER.error(
                "%s - KeyError while data cleaning for train payload: %s", payload.uuid, payload
            )
            responses.append(Response.as_success(_datum.id))
            continue

        if len(train_df) < model_config["win_size"]:
            _LOGGER.info(
                "%s - Skipping training since traindata size: %s is less than winsize: %s",
                payload.uuid,
                train_df.shape,
                win_size,
            )
            responses.append(Response.as_success(_datum.id))
            continue

        x_train, preproc_clf = _preprocess(train_df.to_numpy())
        x_reconerr, model = _train_model(payload.uuid, x_train, model_config)
        thresh_clf = _find_threshold(x_reconerr)

        # TODO change this to just use **composite_keys
        skeys = [payload.composite_keys["namespace"], payload.composite_keys["name"]]

        # TODO catch mlflow exception
        version = save_model(
            skeys=skeys, dkeys=["preproc"], model=preproc_clf, artifact_type="sklearn"
        )
        _LOGGER.info(
            "%s - Preproc model saved with skeys: %s with version: %s", payload.uuid, skeys, version
        )

        version = save_model(skeys=skeys, dkeys=[model_config["model_name"]], model=model)
        _LOGGER.info(
            "%s - Model saved with skeys: %s with version: %s", payload.uuid, skeys, version
        )

        version = save_model(
            skeys=skeys, dkeys=["thresh"], model=thresh_clf, artifact_type="sklearn"
        )
        _LOGGER.info(
            "%s - Threshold model saved with skeys: %s with version: %s",
            payload.uuid,
            skeys,
            version,
        )

        responses.append(Response.as_success(_datum.id))

    return responses
