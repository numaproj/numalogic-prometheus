import os
import time
from typing import Final

import orjson
from numalogic.registry import RedisRegistry, LocalLRUCache
from numalogic.tools.exceptions import RedisRegistryError
from pynumaflow.mapper import Datum


from numaprom import LOGGER
from numaprom.clients.sentinel import get_redis_client
from numaprom.entities import Status, StreamPayload, Header
from numaprom.tools import msg_forward
from numaprom.metrics import increase_redis_conn_error
from numaprom.watcher import ConfigManager

_VERTEX: Final[str] = "preprocess"
AUTH = os.getenv("REDIS_AUTH")
REDIS_CONF = ConfigManager.get_redis_config()
REDIS_CLIENT = get_redis_client(
    REDIS_CONF.host,
    REDIS_CONF.port,
    password=AUTH,
    mastername=REDIS_CONF.master_name,
    recreate=False,
    master_node=False,
)
LOCAL_CACHE_TTL = int(os.getenv("LOCAL_CACHE_TTL", 3600))  # default ttl set to 1 hour


@msg_forward
def preprocess(_: list[str], datum: Datum) -> bytes:
    _start_time = time.perf_counter()
    _in_msg = datum.value.decode("utf-8")

    payload = StreamPayload(**orjson.loads(_in_msg))
    LOGGER.info("{uuid} - Received Payload: {payload} ", uuid=payload.uuid, payload=payload)

    # Load config
    metric_config = ConfigManager.get_metric_config(payload.composite_keys)
    preprocess_cfgs = metric_config.numalogic_conf.preprocess

    # Load preprocess artifact
    local_cache = LocalLRUCache(ttl=LOCAL_CACHE_TTL)
    model_registry = RedisRegistry(client=REDIS_CLIENT, cache_registry=local_cache)

    try:
        preproc_artifact = model_registry.load(
            skeys=[payload.composite_keys["namespace"], payload.composite_keys["name"]],
            dkeys=[_conf.name for _conf in preprocess_cfgs],
        )
    except RedisRegistryError as err:
        LOGGER.exception(
            "{uuid} - Error while fetching preproc artifact, keys: {keys}, err: {err}",
            uuid=payload.uuid,
            keys=payload.composite_keys,
            err=err,
        )
        payload.set_header(Header.STATIC_INFERENCE)
        payload.set_status(Status.RUNTIME_ERROR)
        increase_redis_conn_error(_VERTEX)
        return orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)
    except Exception as ex:
        LOGGER.exception(
            "{uuid} - Unhandled exception while fetching preproc artifact, "
            "keys: {keys}, err: {err}",
            uuid=payload.uuid,
            keys=payload.composite_keys,
            err=ex,
        )
        payload.set_header(Header.STATIC_INFERENCE)
        payload.set_status(Status.RUNTIME_ERROR)
        return orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)

    if not preproc_artifact:
        LOGGER.info(
            "{uuid} - Preprocess artifact not found, forwarding for static thresholding. "
            "Keys: {keys}",
            uuid=payload.uuid,
            keys=payload.composite_keys,
        )
        payload.set_header(Header.STATIC_INFERENCE)
        payload.set_status(Status.ARTIFACT_NOT_FOUND)
        return orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)

    # Perform preprocessing
    x_raw = payload.get_stream_array()
    preproc_clf = preproc_artifact.artifact
    x_scaled = preproc_clf.transform(x_raw)

    # Prepare payload for forwarding
    payload.set_win_arr(x_scaled)
    payload.set_status(Status.PRE_PROCESSED)

    LOGGER.info("{uuid} - Sending Payload: {payload} ", uuid=payload.uuid, payload=payload)
    LOGGER.debug(
        "{uuid} - Time taken in preprocess: {time} sec",
        uuid=payload.uuid,
        time=time.perf_counter() - _start_time,
    )
    return orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)
