import os
import time
import uuid
from typing import List, Tuple, Optional

import numpy as np
import numpy.typing as npt
from orjson import orjson
from pynumaflow.function import Datum
from redis.exceptions import ConnectionError as RedisConnectionError

from numaprom import get_logger
from numaprom.entities import StreamPayload, Status, Header
from numaprom.redis import get_redis_client
from numaprom.tools import msg_forward, create_composite_keys, get_metric_config

_LOGGER = get_logger(__name__)

HOST = os.getenv("REDIS_HOST")
PORT = os.getenv("REDIS_PORT")
AUTH = os.getenv("REDIS_AUTH")


# TODO get the replace value from config
def _clean_arr(
    id_: str,
    ckeys: dict,
    arr: npt.NDArray[float],
    replace_val: float = 0.0,
    inf_replace: float = 1e10,
) -> npt.NDArray[float]:
    if not np.isfinite(arr).any():
        _LOGGER.warning(
            "%s - Non finite values encountered: %s for keys: %s", id_, list(arr), ckeys
        )
    return np.nan_to_num(arr, nan=replace_val, posinf=inf_replace, neginf=-inf_replace)


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

    # Create sliding window
    try:
        elements = __aggregate_window(
            unique_key, msg["timestamp"], value, win_size, buff_size, recreate=False
        )
    except RedisConnectionError:
        _LOGGER.warning("Redis connection failed, recreating the redis client")
        elements = __aggregate_window(
            unique_key, msg["timestamp"], value, win_size, buff_size, recreate=True
        )

    # Drop message if no of elements is less than sequence length needed
    if len(elements) < win_size:
        return None

    # Construct payload object
    _uuid = uuid.uuid4().hex
    composite_keys = create_composite_keys(msg)

    win_list = [float(_val) for _val, _ in elements]
    win_arr = np.asarray(win_list).reshape(-1, 1)
    win_arr = _clean_arr(_uuid, composite_keys, win_arr)

    payload = StreamPayload(
        uuid=uuid.uuid4().hex,
        header=Header.MODEL_INFERENCE,
        composite_keys=composite_keys,
        status=Status.EXTRACTED,
        win_arr=win_arr.copy(),
        win_ts_arr=[str(_ts) for _, _ts in elements],
        metadata=dict(src_labels=msg["labels"]),
    )

    _LOGGER.info("%s - Sending Payload: %r ", payload.uuid, payload)
    _LOGGER.debug(
        "%s - Time taken in window: %.4f sec", payload.uuid, time.perf_counter() - _start_time
    )
    return orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)
