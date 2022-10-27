import os
import logging
import numpy as np
from numalogic.scores import tanh_norm
from pynumaflow.function import Messages, Datum
from redis.exceptions import ConnectionError as RedisConnectionError

from nlogicprom.constants import ARGOCD_METRICS_LIST, ROLLOUTS_METRICS_LIST
from nlogicprom.entities import Payload, Status, PrometheusPayload
from nlogicprom.redis import get_redis_client
from nlogicprom.tools import catch_exception, msgs_forward

LOGGER = logging.getLogger(__name__)

HOST = os.getenv("REDIS_HOST")
PORT = os.getenv("REDIS_PORT")
AUTH = os.getenv("REDIS_AUTH")


def save_to_redis(payload: Payload, recreate: bool):
    r = get_redis_client(HOST, PORT, password=AUTH, recreate=recreate)

    if payload.metric in ARGOCD_METRICS_LIST:
        key = f"{payload.namespace}:{payload.endTS}"
        metrics_list = ARGOCD_METRICS_LIST
    else:
        key = f"{payload.namespace}:{payload.hash_id}:{payload.endTS}"
        metrics_list = ROLLOUTS_METRICS_LIST

    if np.isnan(payload.anomaly):
        r.hset(key, mapping={payload.metric: -1})
    else:
        r.hset(key, mapping={payload.metric: payload.anomaly})

    for m in metrics_list:
        if not r.hexists(name=key, key=m):
            return -1, []

    max_anomaly = -1
    anomalies = []
    for m in metrics_list:
        val = float(r.hget(name=key, key=m))
        anomalies.append(val)
        if max_anomaly < val:
            max_anomaly = val
    r.delete(key)

    return max_anomaly, anomalies


def get_publisher_format(payload: Payload) -> PrometheusPayload:
    if payload.metric in ARGOCD_METRICS_LIST:
        name = f"namespace_app_pod_{payload.metric}_anomaly"
    else:
        name = f"namespace_rollout_{payload.metric}_anomaly"

    prometheus_payload = PrometheusPayload(
        timestamp_ms=int(payload.endTS),
        name=name,
        namespace=payload.namespace,
        subsystem=str(payload.hash_id),
        type="Gauge",
        value=payload.anomaly,
        labels={
            "numalogic": "true",
            "namespace": payload.namespace,
            "model_version": str(payload.model_version),
            "hash_id": str(payload.hash_id),
        },
    )

    return prometheus_payload


def get_unified_format(payload: Payload, max_anomaly: float) -> PrometheusPayload:
    if payload.metric in ARGOCD_METRICS_LIST:
        name = f"namespace_app_pod_unified_anomaly"
    else:
        name = f"namespace_rollout_hash_unified_anomaly"

    prometheus_payload = PrometheusPayload(
        timestamp_ms=int(payload.endTS),
        name=name,
        namespace=payload.namespace,
        subsystem=str(payload.hash_id),
        type="Gauge",
        value=max_anomaly,
        labels={
            "numalogic": "true",
            "namespace": payload.namespace,
            "hash_id": str(payload.hash_id),
        },
    )

    return prometheus_payload


@catch_exception
@msgs_forward
def postprocess(key: str, datum: Datum) -> Messages:
    payload = Payload.from_json(datum.value.decode("utf-8"))

    payload.win_score = payload.get_processed_array()
    payload.anomaly = tanh_norm(np.mean(payload.win_score))
    payload.status = Status.POST_PROCESSED.value
    LOGGER.info("%s - Successfully post-processed payload: %s", payload.uuid, payload.to_json())

    publisher_payload = get_publisher_format(payload)
    publisher_json = publisher_payload.to_json()
    LOGGER.info("%s - Payload sent to publisher: %s", payload.uuid, publisher_json)

    try:
        max_anomaly, anomalies = save_to_redis(payload=payload, recreate=False)
    except RedisConnectionError:
        LOGGER.warning("%s - Redis connection failed, recreating the redis client", payload.uuid)
        max_anomaly, anomalies = save_to_redis(payload=payload, recreate=True)

    if max_anomaly > -1:
        unified_payload = get_unified_format(payload, max_anomaly)
        unified_json = unified_payload.to_json()
        LOGGER.info(
            "%s - Unified anomaly payload sent to publisher: %s", payload.uuid, unified_json
        )
        return [publisher_json, unified_json]

    return [publisher_json]
