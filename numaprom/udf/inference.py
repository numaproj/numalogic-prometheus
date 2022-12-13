import os
import time
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict

from numalogic.registry import MLflowRegistrar
from pynumaflow.function import Messages, Datum

from numaprom.entities import Payload, Status
from numaprom.pipeline import PrometheusPipeline
from numaprom.tools import catch_exception, get_metrics, conditional_forward
from numaprom.constants import (DEFAULT_TRACKING_URI, METRIC_CONFIG)

LOGGER = logging.getLogger(__name__)


# TODO: dont log exception when it is ModelNotFound
def load_model(payload: Payload, model_config: Dict) -> Optional[Dict]:
    tracking_uri = os.getenv("TRACKING_URI", DEFAULT_TRACKING_URI)
    model_name = model_config["MODEL_NAME"]

    try:
        ml_registry = MLflowRegistrar(tracking_uri=tracking_uri)
        artifact_dict = ml_registry.load(
            skeys=model_config["keys"], dkeys=[model_name]
        )
        return artifact_dict
    except Exception as ex:
        LOGGER.exception(
            "%s - Error while loading model from MLflow database. Error:%",
            payload.uuid,
            ex,
        )
        return None


def infer(payload: Payload, artifact_dict: Dict, model_config: Dict) -> Tuple[str, str]:
    win_size = model_config["WIN_SIZE"]
    thresh_min = model_config["THRESHOLD_MIN"]
    model_name = model_config["MODEL_NAME"]

    pipeline = PrometheusPipeline(
        model_plname=model_name,
        model=artifact_dict["primary_artifact"],
        seq_len=win_size,
        threshold_min=thresh_min,
    )
    try:
        pipeline.load_model(
            path_or_buf=None, model=artifact_dict["primary_artifact"], **artifact_dict["metadata"]
        )
        LOGGER.info("%s - Successfully loaded model to pipeline", payload.uuid)
        arr = pipeline.infer(payload.get_processed_array())
        df = payload.get_processed_dataframe()
        df = pd.DataFrame(data=arr, columns=df.columns, index=df.index).reset_index()
        payload.processedMetrics = get_metrics(df)
        payload.status = Status.INFERRED
        payload.std = float(np.mean(pipeline.model_ppl.err_stats["std"]))
        payload.mean = float(np.mean(pipeline.model_ppl.err_stats["mean"]))
        payload.threshold = float(np.mean(pipeline.model_ppl.thresholds))
        payload.model_version = artifact_dict["model_properties"].version
        LOGGER.info("%s - Successfully inferred payload: %s", payload.uuid, payload.to_json())
    except Exception as ex:
        LOGGER.exception(
            "%s - Error while doing inference. Error:%r",
            payload.uuid,
            ex,
        )
        return "", ""
    return "postprocess", payload.to_json()


@catch_exception
@conditional_forward
def inference(key: str, datum: Datum) -> Messages:
    start_inference = time.time()
    payload = Payload.from_json(datum.value.decode("utf-8"))
    model_config = METRIC_CONFIG[payload.metric_name]["model_config"]
    retrain_freq = model_config["RETRAIN_FREQ_HR"]

    LOGGER.info("%s - Starting inference", payload.uuid)

    artifact_dict = load_model(payload, model_config)
    train_payload = json.dumps({"namespace": payload.namespace, "metric": payload.metric})

    if not artifact_dict:
        LOGGER.info("%s - No model found, sending to trainer. Trainer payload: %s", payload.uuid, train_payload)
        return [(model_config["name"], train_payload)]

    LOGGER.info(
        "%s - Successfully loaded model from mlflow, version: %s",
        payload.uuid,
        artifact_dict["model_properties"].version,
    )

    messages = []

    date_updated = artifact_dict["model_properties"].last_updated_timestamp / 1000
    stale_date = (datetime.now() - timedelta(hours=int(retrain_freq))).timestamp()

    if date_updated < stale_date:
        LOGGER.info(
            "%s - Model found is stale, sending to trainer. Trainer payload: %s",
            payload.uuid,
            train_payload,
        )
        messages.append(("train", train_payload))

    messages.append(infer(payload, artifact_dict, model_config))

    if time.time() - start_inference > 5:
        LOGGER.debug(
            "%s - Total time in inference is greater than 5 sec: %s sec",
            payload.uuid,
            time.time() - start_inference,
        )
    else:
        LOGGER.debug(
            "%s - Total time in inference: %s sec",
            payload.uuid,
            time.time() - start_inference,
        )
    return messages
