import os
import json

from pynumaflow.function import Messages, Datum

from numaprom import LOGGER
from numaprom.tools import catch_exception, msg_forward


@catch_exception
@msg_forward
def metric_filter(_: list[str], datum: Datum) -> Messages | None:
    """UDF to filter metrics by labels."""
    LOGGER.debug("Received Msg: {value} ", value=datum.value)

    msg = datum.value.decode("utf-8")
    data = json.loads(msg)

    label = os.getenv("LABEL")
    label_values = json.loads(os.getenv("LABEL_VALUES", "[]"))

    if label in data["labels"] and data["labels"][label] not in label_values:
        return None

    LOGGER.info("Sending Metric: {data} ", data=data)
    return msg
