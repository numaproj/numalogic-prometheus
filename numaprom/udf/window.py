import json
import os
import logging
from typing import List, Tuple
from redis.exceptions import ConnectionError as RedisConnectionError

from pynumaflow.function import Messages, Datum

from numaprom.entities import Metric
from numaprom.redis import get_redis_client
from numaprom.constants import METRIC_CONFIG
from numaprom.tools import decode_msg, msg_forward, catch_exception, extract

_LOGGER = logging.getLogger(__name__)

HOST = os.getenv("REDIS_HOST")
PORT = os.getenv("REDIS_PORT")
AUTH = os.getenv("REDIS_AUTH")


def __aggregate_window(key, ts, value, win_size, buff_size, recreate) -> List[Tuple[str, float]]:
    redis_client = get_redis_client(HOST, PORT, password=AUTH, recreate=recreate)
    with redis_client.pipeline() as pl:
        pl.zadd(key, {f"{value}::{ts}": ts})
        pl.zremrangebyrank(key, -(buff_size + 10), -buff_size)
        pl.zrange(key, -win_size, -1, withscores=True, score_cast_func=int)
        out = pl.execute()
    _window = out[-1]
    _window = list(map(lambda x: (float(x[0].split("::")[0]), x[1]), _window))
    return _window


def get_keys(msg: dict) -> List[str]:
    labels = msg.get("labels")
    metric_name = msg["name"]

    keys = METRIC_CONFIG[metric_name]["keys"]
    values = []
    for k in keys:
        if k in msg:
            values.append(msg[k])
        if k in labels:
            values.append(labels[k])
    return values


@catch_exception
@msg_forward
def window(key: str, datum: Datum) -> Messages:
    msg = decode_msg(datum.value)
    metric_name = msg["name"]

    win_size = METRIC_CONFIG[metric_name]["model_config"]["win_size"]
    buff_size = int(os.getenv("BUFF_SIZE", 10 * win_size))

    if buff_size < win_size:
        raise ValueError(f"Redis list buffer size: {buff_size} is less than window length: {win_size}")

    keys = get_keys(msg)
    unique_key = ":".join(keys)
    value = float(msg["value"])

    try:
        elements = __aggregate_window(
            unique_key, msg["timestamp"], value, win_size, buff_size, recreate=False
        )
    except RedisConnectionError:
        _LOGGER.warning("Redis connection failed, recreating the redis client")
        elements = __aggregate_window(
            unique_key, msg["timestamp"], value, win_size, buff_size, recreate=True
        )

    if len(elements) < win_size:
        return None

    ts_window = [Metric(timestamp=str(_ts), value=float(_val)).to_dict() for _val, _ts in elements]
    msg["window"] = ts_window

    payload = extract(msg, keys)
    payload_json = payload.to_json()
    _LOGGER.info("%s - Extracted payload: %s", payload.uuid, payload_json)
    return payload_json
