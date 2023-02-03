import logging
import os
import time
from collections import OrderedDict

import orjson
from numalogic.registry import MLflowRegistry
from pynumaflow.function import Datum

from numaprom._constants import DEFAULT_TRACKING_URI
from numaprom.entities import Status, StreamPayload, TrainerPayload
from numaprom.tools import msg_forward

_LOGGER = logging.getLogger(__name__)


def _load_artifact(payload: StreamPayload):
    registry = MLflowRegistry(
        tracking_uri=os.getenv("TRACKING_URI", DEFAULT_TRACKING_URI), artifact_type="sklearn"
    )
    return registry.load(
        skeys=[payload.composite_keys["namespace"], payload.composite_keys["name"]],
        dkeys=["preproc"],
    )


@msg_forward
def preprocess(_: str, datum: Datum) -> bytes:
    _start_time = time.perf_counter()
    _in_msg = datum.value.decode("utf-8")
    payload = StreamPayload(**orjson.loads(_in_msg))
    x_raw = payload.get_streamarray()

    _LOGGER.debug("%s - Received Payload: %r ", payload.uuid, payload)

    # Load artifact
    preproc_artifact = _load_artifact(payload)
    if not preproc_artifact:
        _LOGGER.info("%s - Preproc clf not found for %s", payload.uuid, payload.composite_keys)
        train_payload = TrainerPayload(
            uuid=payload.uuid, composite_keys=OrderedDict(payload.composite_keys)
        )
        return orjson.dumps(train_payload)

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
    return orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)
