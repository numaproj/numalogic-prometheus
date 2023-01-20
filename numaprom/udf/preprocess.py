import logging
import time

import orjson
from numalogic.preprocess.transformer import LogTransformer
from pynumaflow.function import Datum

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
    preproc_clf = LogTransformer()
    x_scaled = preproc_clf.transform(x_raw)

    payload.set_win_arr(x_scaled)
    payload.set_status(Status.PRE_PROCESSED)

    _LOGGER.info("%s - Sending Payload: %r ", payload.uuid, payload)
    _LOGGER.debug(
        "%s - Total time to preprocess: %s", payload.uuid, time.perf_counter() - _start_time
    )
    return orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)
