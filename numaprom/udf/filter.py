import json
import logging
import os
from typing import Optional

from pynumaflow.function import Messages, Datum

from numaprom.constants import METRIC_CONFIG
from numaprom.tools import catch_exception, msg_forward

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

    LABEL = os.getenv("LABEL")
    LABEL_VALUES = json.loads(os.getenv("LABEL_VALUES", "[]"))

    if LABEL in data["labels"] and data["labels"][LABEL] not in LABEL_VALUES:
        return None

    LOGGER.info("Sending Metric: %s ", data)
    return msg
