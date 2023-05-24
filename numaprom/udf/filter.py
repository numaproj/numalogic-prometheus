import os
import json
from typing import Optional

from pynumaflow.function import Messages, Datum

from numaprom import get_logger
from numaprom.tools import catch_exception, msg_forward

_LOGGER = get_logger(__name__)


@catch_exception
@msg_forward
def metric_filter(_: list[str], datum: Datum) -> Optional[Messages]:
    """
    UDF to filter metrics by labels
    """
    _LOGGER.debug("Received Msg: %s ", datum.value)

    msg = datum.value.decode("utf-8")
    data = json.loads(msg)

    label = os.getenv("LABEL")
    label_values = json.loads(os.getenv("LABEL_VALUES", "[]"))

    if label in data["labels"] and data["labels"][label] not in label_values:
        return None

    _LOGGER.info("Sending Metric: %s ", data)
    return msg
