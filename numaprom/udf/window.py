import logging
import os
import time
import uuid
from typing import List, Tuple, Optional

import numpy as np
from orjson import orjson
from pynumaflow.function import Datum
from redis.exceptions import ConnectionError as RedisConnectionError

from numaprom.entities import StreamPayload, Status
from numaprom.redis import get_redis_client
from numaprom.tools import msg_forward, create_composite_keys, get_metric_config

_LOGGER = logging.getLogger(__name__)

HOST = os.getenv("REDIS_HOST")
PORT = os.getenv("REDIS_PORT")
AUTH = os.getenv("REDIS_AUTH")


def __aggregate_window(key, ts, value, win_size, buff_size, recreate) -> List[Tuple[float, float]]:
    redis_client = get_redis_client(HOST, PORT, password=AUTH, recreate=recreate)
    with redis_client.pipeline() as pl:
        pl.zadd(key, {f"{value}::{ts}": ts})
        pl.zremrangebyrank(key, -(buff_size + 10), -buff_size)
        pl.zrange(key, -win_size, -1, withscores=True, score_cast_func=int)
        out = pl.execute()
    _window = out[-1]
    _window = list(map(lambda x: (float(x[0].split("::")[0]), x[1]), _window))
    return _window


@msg_forward
def window(_: str, datum: Datum) -> Optional[bytes]:
    """
    UDF to construct windowing of the streaming input data, required by ML models.
    """
    _LOGGER.debug("Received Msg: %s ", datum.value)

    _start_time = time.perf_counter()
    msg = orjson.loads(datum.value)

    metric_name = msg["name"]
    metric_config = get_metric_config(metric_name)
    win_size = metric_config["model_config"]["win_size"]
    buff_size = int(os.getenv("BUFF_SIZE", 10 * win_size))

    if buff_size < win_size:
        raise ValueError(
            f"Redis list buffer size: {buff_size} is less than window length: {win_size}"
        )

    key_map = create_composite_keys(msg)
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

    win_list = [float(_val) for _val, _ in elements]
    payload = StreamPayload(
        uuid=uuid.uuid4().hex,
        composite_keys=create_composite_keys(msg),
        status=Status.EXTRACTED,
        win_arr=np.asarray(win_list).reshape(-1, 1),
        win_ts_arr=[str(_ts) for _, _ts in elements],
        metadata=dict(src_labels=msg["labels"]),
    )

    _LOGGER.info("%s - Sending Payload: %r ", payload.uuid, payload)
    _LOGGER.debug("%s - Total time to window: %s", payload.uuid, time.perf_counter() - _start_time)
    return orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)
