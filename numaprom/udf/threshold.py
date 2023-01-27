import logging
import os
import time

from numalogic.registry import MLflowRegistry
from orjson import orjson
from pynumaflow.function import Datum, Messages, Message

from numaprom._constants import DEFAULT_TRACKING_URI
from numaprom.entities import Status, StreamPayload
from numaprom.tools import get_metric_config

_LOGGER = logging.getLogger(__name__)
_TRAIN_VTX_KEY = "train"
_POSTPROC_VTX_KEY = "postproc"


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
        dkeys=["thresh"],
    )


def threshold(_: str, datum: Datum) -> Messages:
    _start_time = time.perf_counter()
    _in_msg = datum.value.decode("utf-8")
    payload = StreamPayload(**orjson.loads(_in_msg))

    recon_err = payload.get_streamarray()

    # Load config
    metric_config = get_metric_config(payload.composite_keys["name"])
    model_config = metric_config["model_config"]

    # Check if model exists
    thresh_artifact = _load_artifact(payload)
    if not thresh_artifact:
        _LOGGER.info("%s - Thresh clf not found for %s", payload.uuid, payload.composite_keys)
        train_payload = _construct_train_payload(payload, model_config)
        return Messages(Message(key=_TRAIN_VTX_KEY, value=orjson.dumps(train_payload)))

    # Calculate anomaly score
    thresh_clf = thresh_artifact.artifact
    y_score = thresh_clf.predict(recon_err)

    # Prepare payload for forwarding
    payload.set_win_arr(y_score)
    payload.set_status(Status.THRESHOLD)

    _LOGGER.info("%s - Sending Payload: %r ", payload.uuid, payload)
    _LOGGER.debug(
        "%s - Time taken in threshold: %.4f", payload.uuid, time.perf_counter() - _start_time
    )
    return Messages(
        Message(
            key=_POSTPROC_VTX_KEY, value=orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)
        )
    )
