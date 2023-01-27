import logging
import os
import time

import orjson
from numalogic.registry import MLflowRegistry
from pynumaflow.function import Datum, Messages, Message

from numaprom._constants import DEFAULT_TRACKING_URI
from numaprom.entities import Status, StreamPayload
from numaprom.tools import get_metric_config

_LOGGER = logging.getLogger(__name__)
_TRAIN_VTX_KEY = "train"
_INFERENCE_VTX_KEY = "inference"


def _construct_train_payload(payload: StreamPayload, model_config: dict) -> dict:
    return {
        "uuid": payload.uuid,
        **payload.composite_keys,
        "model_config": model_config["name"],
        "resume_training": False,
    }


def _load_artifact(payload: StreamPayload):
    registry = MLflowRegistry(
        tracking_uri=os.getenv("TRACKING_URI", DEFAULT_TRACKING_URI), artifact_type="sklearn"
    )
    return registry.load(
        skeys=[payload.composite_keys["namespace"], payload.composite_keys["name"]],
        dkeys=["preproc"],
    )


def preprocess(_: str, datum: Datum) -> Messages:
    _start_time = time.perf_counter()
    _in_msg = datum.value.decode("utf-8")
    payload = StreamPayload(**orjson.loads(_in_msg))
    x_raw = payload.get_streamarray()

    # Load config
    metric_config = get_metric_config(payload.composite_keys["name"])
    model_config = metric_config["model_config"]

    _LOGGER.debug("%s - Received Payload: %r ", payload.uuid, payload)

    # Load artifact
    preproc_artifact = _load_artifact(payload)
    if not preproc_artifact:
        _LOGGER.info("%s - Preproc clf not found for %s", payload.uuid, payload.composite_keys)
        train_payload = _construct_train_payload(payload, model_config)
        return Messages(Message(key=_TRAIN_VTX_KEY, value=orjson.dumps(train_payload)))

    # Perform preprocessing
    preproc_clf = preproc_artifact.artifact
    x_scaled = preproc_clf.transform(x_raw)

    # Prepare payload for forwarding
    payload.set_win_arr(x_scaled)
    payload.set_status(Status.PRE_PROCESSED)

    _LOGGER.info("%s - Sending Payload: %r ", payload.uuid, payload)
    _LOGGER.debug(
        "%s - Time taken in preprocess: %.4f sec", payload.uuid, time.perf_counter() - _start_time
    )
    return Messages(
        Message(
            key=_INFERENCE_VTX_KEY, value=orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)
        )
    )
