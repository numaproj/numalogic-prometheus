import os
import sys
import time
import json
import logging
import multiprocessing
from typing import Optional, Dict
from mlflow.entities.model_registry import ModelVersion

from numalogic.models.autoencoder.variants import VanillaAE, Conv1dAE, LSTMAE
from numalogic.preprocess.transformer import LogTransformer

from numaprom.constants import (
    DEFAULT_PROMETHEUS_SERVER,
    METRIC_CONFIG,
)
from numaprom.pipeline import PrometheusPipeline
from numaprom.tools import load_model, save_model

LOGGER = logging.getLogger(__name__)


def rollout_trainer(payload: Dict) -> Optional[ModelVersion]:
    start_train = time.time()

    namespace = payload["namespace"]
    metric_name = payload["name"]
    resume_training = payload["resume_training"]

    metric_config = METRIC_CONFIG[metric_name]
    model_config = metric_config["model_config"]

    LOGGER.info("Starting training for namespace: %s, metric: %s", namespace, metric_name)

    pipeline = PrometheusPipeline(
        preprocess_steps=[LogTransformer()],
        model=LSTMAE(seq_len=model_config["win_size"], no_features=1, embedding_dim=64),
        model_plname=model_config["model_name"],
        seq_len=model_config["win_size"],
        threshold_min=model_config["threshold_min"],
        num_epochs=model_config["num_epochs"]
    )

    if resume_training:
        artifact_from_registry = load_model(skeys=[namespace, metric_name], dkeys=[model_config["model_name"]])
        if artifact_from_registry:
            pipeline.load_model(model=artifact_from_registry["primary_artifact"])
            LOGGER.info("Resume training for namespace: %s, metric: %s", namespace, metric_name)
    else:
        LOGGER.info("Training for namespace: %s, metric: %s", namespace, metric_name)

    prometheus_server = os.getenv("PROMETHEUS_SERVER", DEFAULT_PROMETHEUS_SERVER)
    df = pipeline.fetch_data(delta_hr=4, prometheus_server=prometheus_server,
                             metric_name=metric_name, labels_map={"namespace": namespace}, return_labels=["hash_id"])
    df = pipeline.clean_rollout_data(df)
    LOGGER.info("Time taken to fetch data for rollout: %s", time.time() - start_train)

    # Todo: What if we do not have enough datapoints?
    if df.empty:
        return None

    x_scaled = pipeline.preprocess(df.to_numpy())
    pipeline.train(x_scaled)
    LOGGER.info("Time taken to train model for rollout: %s", time.time() - start_train)

    version = save_model(
        skeys=[namespace, metric_name],
        dkeys=[model_config["model_name"]],
        model=pipeline.model,
        **pipeline.model_ppl.model_properties
    )
    LOGGER.info("Total time in trainer: %s", time.time() - start_train)

    return version


# TODO: retry logic if model training fails
def argocd_trainer(payload: Dict) -> Optional[ModelVersion]:
    start_train = time.time()

    namespace = payload["namespace"]
    metric_name = payload["name"]

    metric_config = METRIC_CONFIG[metric_name]
    model_config = metric_config["model_config"]

    LOGGER.info("Starting training for namespace: %s, metric: %s", namespace, metric_name)

    model = VanillaAE(model_config["win_size"])
    if metric_config["model"] == "Conv1dAE":
        model = Conv1dAE(in_channels=1, enc_channels=8)

    pipeline = PrometheusPipeline(
        preprocess_steps=[LogTransformer()],
        model_plname=model_config["model_name"],
        model=model,
        seq_len=model_config["win_size"],
        threshold_min=model_config["threshold_min"],
        num_epochs=model_config["num_epochs"]
    )

    prometheus_server = os.getenv("PROMETHEUS_SERVER", DEFAULT_PROMETHEUS_SERVER)
    df = pipeline.fetch_data(delta_hr=15, prometheus_server=prometheus_server,
                             metric_name=metric_name, labels_map={"namespace": namespace})
    df = pipeline.clean_data(df)
    LOGGER.info("Time taken to fetch data: %s", time.time() - start_train)

    if len(df) < model_config["win_size"]:
        return None

    x_scaled = pipeline.preprocess(df.to_numpy())
    pipeline.train(x_scaled)
    LOGGER.info("Time taken to train model: %s", time.time() - start_train)

    version = save_model(
        skeys=[namespace, metric_name],
        dkeys=[model_config["model_name"]],
        model=pipeline.model,
        **pipeline.model_ppl.model_properties
    )
    LOGGER.info("Total time in trainer: %s", time.time() - start_train)

    return version


def train(payload: str) -> Optional[ModelVersion]:
    payload_json = json.loads(payload)

    metric_config = METRIC_CONFIG[payload_json["key_map"]["name"]]
    model_config = metric_config["model_config"]

    if model_config["name"] == "argo_cd":
        return argocd_trainer(payload_json)
    elif model_config["name"] == "argo_rollouts":
        return rollout_trainer(payload_json)
    else:
        raise NotImplementedError("Trainer not implemented for %s", model_config["name"])


if __name__ == "__main__":
    payloads = json.loads(sys.argv[1])
    LOGGER.info("Payloads received for training: %s", payloads)

    try:
        num_processes = int(sys.argv[2])
        pool = multiprocessing.Pool(num_processes)
    except IndexError:
        pool = multiprocessing.Pool()

    pool.map(train, payloads)
