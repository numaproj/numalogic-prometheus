import json
from datetime import datetime
from typing import AsyncIterable, List

from pynumaflow.function import Message, Datum, Metadata, Messages

from numaprom import get_logger

_LOGGER = get_logger(__name__)


async def window(keys: List[str], datums: AsyncIterable[Datum], md: Metadata) -> Messages:
    _LOGGER.info("Received Msg: { keys: %s, md: %s}", keys, md)

    start_time = datetime.timestamp(md.interval_window.end)
    end_time = datetime.timestamp(md.interval_window.end)

    data = []
    metadata = {}
    async for d in datums:
        json_obj = json.loads(d.value)
        _LOGGER.info("Processing Msg: { keys: %s, value: %s }", keys, json_obj)
        data.append({
            "timestamp": json_obj["timestamp"],
            json_obj["name"]: json_obj["value"]
        })
        metadata = json_obj["labels"]

    output = {
        "start_time": start_time,
        "end_time": end_time,
        "data": data,
        "metadata": metadata
    }

    json_str = json.dumps(output)
    _LOGGER.info("Sending Msg: { keys: %s, value: %s }", keys, json_str)
    return Messages(Message(str.encode(json_str), keys=keys))


