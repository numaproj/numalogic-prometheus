import logging
import os
import time

import orjson
from numalogic.registry import MLflowRegistry
from pynumaflow.function import Datum

from numaprom._constants import DEFAULT_TRACKING_URI
from numaprom.entities import Status, StreamPayload
from numaprom.tools import msg_forward

_LOGGER = logging.getLogger(__name__)


@msg_forward
def preprocess(_: str, datum: Datum) -> bytes:
    _start_time = time.perf_counter()
    _in_msg = datum.value.decode("utf-8")
    payload = StreamPayload(**orjson.loads(_in_msg))

    _LOGGER.debug("%s - Received Payload: %r ", payload.uuid, payload)

    x_raw = payload.get_streamarray()

    registry = MLflowRegistry(
        tracking_uri=os.getenv("TRACKING_URI", DEFAULT_TRACKING_URI), artifact_type="sklearn"
    )
    preproc_artifact = registry.load(
        skeys=[payload.composite_keys["namespace"], payload.composite_keys["name"]],
        dkeys=["preproc"],
    )
    if not preproc_artifact:
        payload.set_status(Status.ARTIFACT_NOT_FOUND)
        _LOGGER.info("%s - Preproc clf not found for %s", payload.uuid, payload.composite_keys)
        return orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)

    preproc_clf = preproc_artifact.artifact
    x_scaled = preproc_clf.transform(x_raw)

    payload.set_win_arr(x_scaled)
    payload.set_status(Status.PRE_PROCESSED)

    _LOGGER.info("%s - Sending Payload: %r ", payload.uuid, payload)
    _LOGGER.debug(
        "%s - Time taken in preprocess: %.4f", payload.uuid, time.perf_counter() - _start_time
    )
    return orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)
