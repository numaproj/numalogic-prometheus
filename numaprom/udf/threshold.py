import os
import time
from collections import OrderedDict
from typing import Final

from numalogic.registry import RedisRegistry, LocalLRUCache
from numalogic.tools.exceptions import RedisRegistryError
from orjson import orjson
from pynumaflow.mapper import Datum

from numaprom import LOGGER
from numaprom._constants import TRAIN_VTX_KEY, POSTPROC_VTX_KEY
from numaprom.clients.sentinel import get_redis_client_from_conf
from numaprom.entities import Status, TrainerPayload, PayloadFactory, Header
from numaprom.metrics import increase_redis_conn_error
from numaprom.tools import conditional_forward, calculate_static_thresh
from numaprom.watcher import ConfigManager


_VERTEX: Final[str] = "threshold"
LOCAL_CACHE_TTL = int(os.getenv("LOCAL_CACHE_TTL", 3600))  # default ttl set to 1 hour


def _get_static_thresh_payload(payload, metric_config) -> bytes:
    """Calculates static thresholding, and returns a serialized json bytes payload."""
    static_scores = calculate_static_thresh(payload, metric_config.static_threshold)

    payload.set_win_arr(static_scores)
    payload.set_header(Header.STATIC_INFERENCE)
    payload.set_status(Status.ARTIFACT_NOT_FOUND)
    payload.set_metadata("version", -1)

    LOGGER.info(
        "{uuid} - Static thresholding complete for payload: {payload}",
        uuid=payload.uuid,
        payload=payload,
    )
    return orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)


@conditional_forward
def threshold(_: list[str], datum: Datum) -> list[tuple[str, bytes]]:
    _start_time = time.perf_counter()
    _in_msg = datum.value.decode("utf-8")

    # Construct payload objects
    payload = PayloadFactory.from_json(_in_msg)
    train_payload = TrainerPayload(
        uuid=payload.uuid, composite_keys=OrderedDict(payload.composite_keys)
    )

    # Load config
    metric_config = ConfigManager.get_metric_config(payload.composite_keys)
    thresh_cfg = metric_config.numalogic_conf.threshold

    # Check if payload needs static inference
    if payload.header == Header.STATIC_INFERENCE:
        LOGGER.info(
            "{uuid} - Sending to trainer and performing static thresholding. Keys: {keys}",
            uuid=payload.uuid,
            keys=payload.composite_keys,
        )
        return [
            (TRAIN_VTX_KEY, orjson.dumps(train_payload)),
            (POSTPROC_VTX_KEY, _get_static_thresh_payload(payload, metric_config)),
        ]

    # load threshold artifact
    local_cache = LocalLRUCache(ttl=LOCAL_CACHE_TTL)
    model_registry = RedisRegistry(
        client=get_redis_client_from_conf(master_node=False), cache_registry=local_cache
    )
    try:
        thresh_artifact = model_registry.load(
            skeys=[payload.composite_keys["namespace"], payload.composite_keys["name"]],
            dkeys=[thresh_cfg.name],
        )
    except RedisRegistryError as err:
        LOGGER.exception(
            "{uuid} - Error while fetching threshold artifact, keys: {keys}, err: {err}",
            uuid=payload.uuid,
            keys=payload.composite_keys,
            err=err,
        )
        increase_redis_conn_error(_VERTEX)
        payload.set_header(Header.STATIC_INFERENCE)
        payload.set_status(Status.RUNTIME_ERROR)
        return [
            (TRAIN_VTX_KEY, orjson.dumps(train_payload)),
            (POSTPROC_VTX_KEY, _get_static_thresh_payload(payload, metric_config)),
        ]
    except Exception as ex:
        LOGGER.exception(
            "{uuid} - Unhandled exception while fetching threshold artifact, "
            "keys: {keys}, err: {err}",
            uuid=payload.uuid,
            keys=payload.composite_keys,
            err=ex,
        )
        payload.set_header(Header.STATIC_INFERENCE)
        payload.set_status(Status.RUNTIME_ERROR)
        return [
            (TRAIN_VTX_KEY, orjson.dumps(train_payload)),
            (POSTPROC_VTX_KEY, _get_static_thresh_payload(payload, metric_config)),
        ]

    if not thresh_artifact:
        LOGGER.info(
            "{uuid} - Threshold artifact not found, performing static thresholding. Keys: {keys}",
            uuid=payload.uuid,
            keys=payload.composite_keys,
        )
        payload.set_header(Header.STATIC_INFERENCE)
        payload.set_status(Status.ARTIFACT_NOT_FOUND)
        return [
            (TRAIN_VTX_KEY, orjson.dumps(train_payload)),
            (POSTPROC_VTX_KEY, _get_static_thresh_payload(payload, metric_config)),
        ]

    messages = []
    if payload.header == Header.MODEL_STALE:
        messages.append((TRAIN_VTX_KEY, orjson.dumps(train_payload)))

    # Calculate anomaly score
    recon_err = payload.get_stream_array()
    thresh_clf = thresh_artifact.artifact
    y_score = thresh_clf.score_samples(recon_err)

    # Prepare payload for forwarding
    payload.set_win_arr(y_score)
    payload.set_status(Status.THRESHOLD)
    messages.append((POSTPROC_VTX_KEY, orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)))

    LOGGER.info("{uuid} - Sending Payload: {payload} ", uuid=payload.uuid, payload=payload)
    LOGGER.debug(
        "{uuid} - Time taken in threshold: {time} sec",
        uuid=payload.uuid,
        time=time.perf_counter() - _start_time,
    )
    return messages
