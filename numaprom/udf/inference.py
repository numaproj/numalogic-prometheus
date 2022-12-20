import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Dict

from pynumaflow.function import Messages, Datum

from numaprom.entities import Payload, Status
from numaprom.pipeline import PrometheusPipeline
from numaprom.tools import (
    catch_exception,
    get_metrics,
    conditional_forward,
    load_model,
    get_metric_config,
)

LOGGER = logging.getLogger(__name__)


def infer(payload: Payload, artifact_dict: Dict, model_config: Dict) -> Tuple[str, str]:
    pipeline = PrometheusPipeline(
        model_plname=model_config["model_name"],
        model=artifact_dict["primary_artifact"],
        seq_len=model_config["win_size"],
        threshold_min=model_config["threshold_min"],
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

    metric_config = get_metric_config(payload.metric_name)
    model_config = metric_config["model_config"]

    LOGGER.info("%s - Starting inference", payload.uuid)

    artifact_dict = load_model(
        skeys=[payload.key_map["namespace"], payload.key_map["name"]],
        dkeys=[model_config["model_name"]],
    )

    train_payload = payload.key_map
    train_payload["model_config"] = model_config["name"]

    if not artifact_dict:
        train_payload["resume_training"] = False
        LOGGER.info(
            "%s - No model found, sending to trainer. Trainer payload: %s",
            payload.uuid,
            train_payload,
        )
        return [("train", json.dumps(train_payload))]

    LOGGER.info(
        "%s - Successfully loaded model from mlflow, version: %s",
        payload.uuid,
        artifact_dict["model_properties"].version,
    )

    messages = []

    date_updated = artifact_dict["model_properties"].last_updated_timestamp / 1000
    stale_date = (
        datetime.now() - timedelta(hours=int(model_config["retrain_freq_hr"]))
    ).timestamp()

    if date_updated < stale_date:
        train_payload["resume_training"] = True
        LOGGER.info(
            "%s - Model found is stale, sending to trainer. Trainer payload: %s",
            payload.uuid,
            train_payload,
        )
        messages.append(("train", json.dumps(train_payload)))

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
