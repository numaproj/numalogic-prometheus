import json
import os
import logging
from typing import List, Tuple
from redis.exceptions import ConnectionError as RedisConnectionError

from pynumaflow.function import Messages, Datum

from numaprom.entities import Metric
from numaprom.redis import get_redis_client
from numaprom.tools import msg_forward, catch_exception, extract, get_key_map, get_metric_config

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


@catch_exception
@msg_forward
def window(key: str, datum: Datum) -> Messages:
    msg = datum.value.decode("utf-8")

    try:
        msg = json.loads(msg)
    except Exception as ex:
        _LOGGER.exception("Error in Json serialization: %r", ex)
        return None

    metric_name = msg["name"]

    metric_config = get_metric_config(metric_name)
    win_size = metric_config["model_config"]["win_size"]
    buff_size = int(os.getenv("BUFF_SIZE", 10 * win_size))

    if buff_size < win_size:
        raise ValueError(
            f"Redis list buffer size: {buff_size} is less than window length: {win_size}"
        )

    key_map = get_key_map(msg)
    unique_key = ":".join(key_map.values())
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

    payload = extract(msg)
    payload_json = payload.to_json()
    return payload_json
