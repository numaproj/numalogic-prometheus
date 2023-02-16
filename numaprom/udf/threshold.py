import time
from orjson import orjson
from collections import OrderedDict

from pynumaflow.function import Datum

from numaprom import get_logger
from numaprom._constants import TRAIN_VTX_KEY, POSTPROC_VTX_KEY
from numaprom.entities import Status, TrainerPayload, PayloadFactory, Header
from numaprom.tools import (
    conditional_forward,
    calculate_static_thresh,
    load_model,
    get_metric_config,
)

_LOGGER = get_logger(__name__)


@conditional_forward
def threshold(_: str, datum: Datum) -> list[tuple[str, bytes]]:
    _start_time = time.perf_counter()
    _in_msg = datum.value.decode("utf-8")

    # Construct payload objects
    payload = PayloadFactory.from_json(_in_msg)
    train_payload = TrainerPayload(
        uuid=payload.uuid, composite_keys=OrderedDict(payload.composite_keys)
    )

    # Load config
    metric_config = get_metric_config(payload.composite_keys["name"])

    # Check if payload needs static inference
    if payload.header == Header.STATIC_INFERENCE:
        _LOGGER.debug(
            "%s - Models not found in the previous steps, performing static thresholding. Keys: %s",
            payload.uuid,
            payload.composite_keys,
        )
        return [
            (TRAIN_VTX_KEY, orjson.dumps(train_payload)),
            (POSTPROC_VTX_KEY, calculate_static_thresh(payload, metric_config["static_threshold"])),
        ]

    # load threshold artifact
    thresh_artifact = load_model(
        skeys=[payload.composite_keys["namespace"], payload.composite_keys["name"]],
        dkeys=["thresh"],
        artifact_type="sklearn",
    )
    if not thresh_artifact:
        _LOGGER.info(
            "%s - Threshold artifact not found, performing static thresholding. Keys: %s",
            payload.uuid,
            payload.composite_keys,
        )
        payload.set_header(Header.STATIC_INFERENCE)
        payload.set_status(Status.ARTIFACT_NOT_FOUND)
        return [
            (TRAIN_VTX_KEY, orjson.dumps(train_payload)),
            (POSTPROC_VTX_KEY, calculate_static_thresh(payload, metric_config["static_threshold"])),
        ]

    messages = []
    if payload.header == Header.MODEL_STALE:
        messages.append((TRAIN_VTX_KEY, orjson.dumps(train_payload)))

    # Calculate anomaly score
    recon_err = payload.get_stream_array()
    thresh_clf = thresh_artifact.artifact
    y_score = thresh_clf.score_samples(recon_err)

    # Prepare payload for forwarding
    payload.set_win_arr(y_score)
    payload.set_status(Status.THRESHOLD)
    messages.append((POSTPROC_VTX_KEY, orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)))

    _LOGGER.info("%s - Sending Payload: %r ", payload.uuid, payload)
    _LOGGER.debug(
        "%s - Time taken in threshold: %.4f", payload.uuid, time.perf_counter() - _start_time
    )
    return messages
