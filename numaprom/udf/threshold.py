import logging
import os
import time
from collections import OrderedDict

from numalogic.registry import MLflowRegistry
from orjson import orjson
from pynumaflow.function import Datum

from numaprom._constants import DEFAULT_TRACKING_URI, TRAIN_VTX_KEY, POSTPROC_VTX_KEY
from numaprom.entities import Status, StreamPayload, TrainerPayload, PayloadFactory, Header
from numaprom.tools import conditional_forward, calculate_static_thresh

_LOGGER = logging.getLogger(__name__)
STATIC_LIMIT: float = float(os.getenv("STATIC_LIMIT", 3.0))


def _load_artifact(payload: StreamPayload):
    registry = MLflowRegistry(
        tracking_uri=os.getenv("TRACKING_URI", DEFAULT_TRACKING_URI), artifact_type="sklearn"
    )
    return registry.load(
        skeys=[payload.composite_keys["namespace"], payload.composite_keys["name"]],
        dkeys=["thresh"],
    )


@conditional_forward
def threshold(_: str, datum: Datum) -> list[tuple[str, bytes]]:
    _start_time = time.perf_counter()
    _in_msg = datum.value.decode("utf-8")

    # Construct payload object
    payload = PayloadFactory.from_json(_in_msg)

    # Check if trainer payload is passed on from previous vtx
    if isinstance(payload, TrainerPayload):
        _LOGGER.debug("%s - Previous clf not found. Sending to trainer..", payload.uuid)
        return [(TRAIN_VTX_KEY, orjson.dumps(payload))]

    # Check if this payload has performed static thresholding
    if payload.header == Header.STATIC_INFERENCE:
        _LOGGER.debug("%s - Relaying forward static threshold payload")
        return [orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)]

    recon_err = payload.get_stream_array()

    # Check if model exists
    thresh_artifact = _load_artifact(payload)
    if not thresh_artifact:
        msgs = []
        _LOGGER.info(
            "%s - Thresh clf not found for %s. Sending to trainer..",
            payload.uuid,
            payload.composite_keys,
        )
        train_payload = TrainerPayload(
            uuid=payload.uuid, composite_keys=OrderedDict(payload.composite_keys)
        )
        msgs.append((TRAIN_VTX_KEY, orjson.dumps(train_payload)))

        # Calculate scores using static threshold
        msgs.append((POSTPROC_VTX_KEY, calculate_static_thresh(payload, STATIC_LIMIT)))
        return msgs

    # Calculate anomaly score
    thresh_clf = thresh_artifact.artifact
    y_score = thresh_clf.score_samples(recon_err)

    # Prepare payload for forwarding
    payload.set_win_arr(y_score)
    payload.set_status(Status.THRESHOLD)

    _LOGGER.info("%s - Sending Payload: %r ", payload.uuid, payload)
    _LOGGER.debug(
        "%s - Time taken in threshold: %.4f", payload.uuid, time.perf_counter() - _start_time
    )
    return [(POSTPROC_VTX_KEY, orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY))]
