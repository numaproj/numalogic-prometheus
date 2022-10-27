import os
import sys
import time
import json
import mlflow
import logging
import multiprocessing
from typing import Optional, Sequence, Dict
from mlflow.entities.model_registry import ModelVersion

from numalogic.models.autoencoder.variants import VanillaAE, Conv1dAE, LSTMAE
from numalogic.preprocess.transformer import LogTransformer
from numalogic.registry import MLflowRegistrar

from nlogicprom.constants import (
    DEFAULT_WIN_SIZE,
    DEFAULT_THRESHOLD_MIN,
    DEFAULT_TRACKING_URI,
    DEFAULT_MODEL_NAME,
    DEFAULT_PROMETHEUS_SERVER,
    DEFAULT_ROLLOUT_WIN_SIZE,
    DEFAULT_ROLLOUT_MODEL_NAME,
    DEFAULT_ROLLOUT_THRESHOLD_MIN,
    ARGO_CD,
    ARGO_ROLLOUTS,
    ROLLOUTS_METRICS_LIST,
    DEFAULT_RESUME_TRAINING,
)
from nlogicprom.entities import MetricType
from nlogicprom.pipeline import PrometheusPipeline

LOGGER = logging.getLogger(__name__)


def load_model(payload) -> Optional[Dict]:
    try:
        if payload["metric"] in ROLLOUTS_METRICS_LIST:
            model_name = os.getenv("ROLLOUT_MODEL_NAME", DEFAULT_ROLLOUT_MODEL_NAME)
        else:
            model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)

        tracking_uri = os.getenv("TRACKING_URI", DEFAULT_TRACKING_URI)
        ml_registry = MLflowRegistrar(tracking_uri=tracking_uri)
        artifact_dict = ml_registry.load(
            skeys=[payload["namespace"], payload["metric"]], dkeys=[model_name]
        )
        return artifact_dict
    except Exception as ex:
        LOGGER.error("Error while loading model from MLflow database: %s", ex)
        return None


def save_model(
        skeys: Sequence[str], dkeys: Sequence[str], model, **metadata
) -> Optional[ModelVersion]:
    tracking_uri = os.getenv("TRACKING_URI", DEFAULT_TRACKING_URI)
    ml_registry = MLflowRegistrar(tracking_uri=tracking_uri, artifact_type="pytorch")
    mlflow.start_run()
    version = ml_registry.save(skeys=skeys, dkeys=dkeys, primary_artifact=model, **metadata)
    LOGGER.info("Successfully saved the model to mlflow. Model version: %s", version)
    mlflow.end_run()
    return version


def rollout_train(rollout_payload: str) -> Optional[ModelVersion]:
    start_train = time.time()
    rollout_payload_json = json.loads(rollout_payload)
    namespace = rollout_payload_json["namespace"]
    metric = rollout_payload_json["metric"]

    LOGGER.info("Starting training for namespace: %s, metric: %s", namespace, metric)

    rollout_win_size = int(os.getenv("ROLLOUT_WIN_SIZE", DEFAULT_ROLLOUT_WIN_SIZE))
    rollout_thresh_min = float(os.getenv("ROLLOUT_THRESHOLD_MIN", DEFAULT_ROLLOUT_THRESHOLD_MIN))
    rollout_model_name = os.getenv("ROLLOUT_MODEL_NAME", DEFAULT_ROLLOUT_MODEL_NAME)
    prometheus_server = os.getenv("PROMETHEUS_SERVER", DEFAULT_PROMETHEUS_SERVER)
    resume_training = os.getenv("RESUME_TRAINING", DEFAULT_RESUME_TRAINING)

    artifact_from_registry = load_model(rollout_payload_json)
    pipeline = PrometheusPipeline(
        namespace=namespace,
        metric=metric,
        preprocess_steps=[LogTransformer()],
        model=LSTMAE(seq_len=rollout_win_size, no_features=1, embedding_dim=64),
        model_plname=rollout_model_name,
        seq_len=rollout_win_size,
        threshold_min=rollout_thresh_min,
        num_epochs=50,
    )

    if resume_training == "True" and artifact_from_registry:
        pipeline.load_model(model=artifact_from_registry["primary_artifact"])
        LOGGER.info("Resume training for namespace: %s, metric: %s", namespace, metric)
    else:
        LOGGER.info("Training for namespace: %s, metric: %s", namespace, metric)

    df = pipeline.fetch_data(delta_hr=4, prometheus_server=prometheus_server, hash_col=True)
    df = pipeline.clean_rollout_data(df)
    LOGGER.info("Time taken to fetch data for rollout: %s",time.time() - start_train)

    # Todo: What if we do not have enough datapoints?
    if df.empty:
        return None

    x_scaled = pipeline.preprocess(df.to_numpy())
    pipeline.train(x_scaled)
    LOGGER.info("Time taken to train model for rollout: %s",time.time() - start_train)

    version = save_model(
        skeys=[namespace, metric],
        dkeys=[rollout_model_name],
        model=pipeline.model,
        **pipeline.model_ppl.model_properties
    )
    LOGGER.info("Total time in trainer: %s", time.time() - start_train)

    return version


# TODO: retry logic if model training fails
def train(payload: str) -> Optional[ModelVersion]:
    payload_json = json.loads(payload)
    namespace = payload_json["namespace"]
    metric = payload_json["metric"]

    LOGGER.info("Starting training for namespace: %s, metric: %s", namespace, metric)
    start_train = time.time()

    win_size = int(os.getenv("WIN_SIZE", DEFAULT_WIN_SIZE))
    thresh_min = float(os.getenv("THRESHOLD_MIN", DEFAULT_THRESHOLD_MIN))
    model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
    prometheus_server = os.getenv("PROMETHEUS_SERVER", DEFAULT_PROMETHEUS_SERVER)

    model = VanillaAE(win_size)
    if metric == MetricType.LATENCY.value:
        model = Conv1dAE(in_channels=1, enc_channels=8)

    pipeline = PrometheusPipeline(
        namespace=namespace,
        metric=metric,
        preprocess_steps=[LogTransformer()],
        model_plname=model_name,
        model=model,
        seq_len=win_size,
        threshold_min=thresh_min,
    )

    df = pipeline.fetch_data(delta_hr=15, prometheus_server=prometheus_server)
    df = pipeline.clean_data(df)
    LOGGER.info("Time taken to fetch data: %s", time.time() - start_train)

    if len(df) < win_size:
        return None

    x_scaled = pipeline.preprocess(df.to_numpy())
    pipeline.train(x_scaled)
    LOGGER.info("Time taken to train model: %s", time.time() - start_train)

    version = save_model(
        skeys=[namespace, metric],
        dkeys=[model_name],
        model=pipeline.model,
        **pipeline.model_ppl.model_properties
    )
    LOGGER.info("Total time in trainer: %s", time.time() - start_train)

    return version


if __name__ == "__main__":
    payloads = json.loads(sys.argv[1])
    LOGGER.info("Payloads received for training: %s", payloads)

    try:
        num_processes = int(sys.argv[2])
        pool = multiprocessing.Pool(num_processes)
    except IndexError:
        pool = multiprocessing.Pool()

    trainer = os.getenv("TRAINER")

    if trainer == ARGO_CD:
        pool.map(train, payloads)
    elif trainer == ARGO_ROLLOUTS:
        pool.map(rollout_train, payloads)
