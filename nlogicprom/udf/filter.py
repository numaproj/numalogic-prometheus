import json
import logging
from typing import Optional

from pynumaflow.function import Messages, Datum

from nlogicprom.constants import METRICS
from nlogicprom.tools import catch_exception, msg_forward

LOGGER = logging.getLogger(__name__)


@catch_exception
@msg_forward
def metric_filter(key: str, datum: Datum) -> Optional[Messages]:
    msg = datum.value.decode("utf-8")

    try:
        data = json.loads(msg)
    except Exception as ex:
        LOGGER.exception("Error in Json serialization: %r", ex)
        return None

    if data["name"] not in METRICS:
        return None

    LOGGER.info("Sending Metric: %s ", data)
    return msg
