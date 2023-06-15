import json
from typing import List

from pynumaflow.function import Datum, Message, Messages

from numaprom import get_logger

_LOGGER = get_logger(__name__)


def keying(keys: List[str], datum: Datum) -> Messages:
    _ = datum.event_time
    _ = datum.watermark
    messages = Messages()
    _LOGGER.info("Received Msg: { keys: %s, value: %s }", datum.value, keys)

    try:
        json_obj = json.loads(datum.value)
    except Exception as e:
        _LOGGER.error("Error while reading input json %r", e)
        messages.append(Message.to_drop())
        return messages

    metric_name = json_obj["name"]
    namespace = json_obj["labels"]["namespace"]
    app = json_obj["labels"]["app"]
    rollouts_pod_template_hash = json_obj["labels"]["rollouts_pod_template_hash"]

    keys = ["default-argorollouts", namespace, metric_name, app, rollouts_pod_template_hash]
    messages.append(Message(datum.value, keys=keys))
    _LOGGER.info("Sending Msgs: { keys: %s value: %s }", keys, datum.value)
    return messages
