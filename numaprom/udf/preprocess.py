import time
import orjson

from pynumaflow.function import Datum

from numaprom import get_logger
from numaprom.entities import Status, StreamPayload, Header
from numaprom.tools import msg_forward, load_model

_LOGGER = get_logger(__name__)


@msg_forward
def preprocess(_: str, datum: Datum) -> bytes:
    _start_time = time.perf_counter()
    _in_msg = datum.value.decode("utf-8")

    payload = StreamPayload(**orjson.loads(_in_msg))
    _LOGGER.info("%s - Received Payload: %r ", payload.uuid, payload)

    # Load preprocess artifact
    preproc_artifact = load_model(
        skeys=[payload.composite_keys["namespace"], payload.composite_keys["name"]],
        dkeys=["preproc"],
        artifact_type="sklearn",
    )
    if not preproc_artifact:
        _LOGGER.info(
            "%s - Preprocess artifact not found, forwarding for static thresholding. Keys: %s",
            payload.uuid,
            payload.composite_keys,
        )
        payload.set_header(Header.STATIC_INFERENCE)
        payload.set_status(Status.ARTIFACT_NOT_FOUND)
        return orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)

    # Perform preprocessing
    x_raw = payload.get_stream_array()
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
