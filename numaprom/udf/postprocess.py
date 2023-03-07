import os
import time
import numpy as np
from numalogic.config import PostprocessFactory
from orjson import orjson
from typing import List
from redis.exceptions import ConnectionError as RedisConnectionError

from pynumaflow.function import Datum

from numaprom import get_logger
from numaprom.config._config import UnifiedConf
from numaprom.entities import Status, PrometheusPayload, StreamPayload
from numaprom.redis import get_redis_client
from numaprom.tools import msgs_forward, get_unified_config, get_metric_config

_LOGGER = get_logger(__name__)

HOST = os.getenv("REDIS_HOST")
PORT = os.getenv("REDIS_PORT")
AUTH = os.getenv("REDIS_AUTH")


def save_to_redis(
    payload: StreamPayload, final_score: float, recreate: bool, unified_config: UnifiedConf
):
    r = get_redis_client(HOST, PORT, password=AUTH, recreate=recreate)

    metric_name = payload.composite_keys["name"]

    r_keys = payload.composite_keys
    r_keys.pop("name")

    metrics_list = unified_config.unified_metrics
    r_key = f"{':'.join(r_keys.values())}:{payload.end_ts}"

    final_score = -1 if np.isnan(final_score) else final_score
    r.hset(r_key, mapping={metric_name: final_score})
    _LOGGER.debug(
        "%s - Saved to redis, redis_key: %s, metric: %s, anomaly_score: %d",
        payload.uuid,
        r_key,
        metric_name,
        final_score,
    )

    for m in metrics_list:
        if not r.hexists(name=r_key, key=m):
            _LOGGER.debug(
                "%s - Unable to generate unified anomaly, missing metric: %s, redis_key: %s",
                payload.uuid,
                m,
                r_key,
            )
            return -1, []

    _LOGGER.debug("%s - Received all metrics, generating unified anomaly", payload.uuid)

    unified_anomaly = -1
    anomalies = []

    unified_weights = unified_config.unified_weights
    if unified_weights:
        _LOGGER.debug("%s - Generating unified anomaly, using unified weights", payload.uuid)
        for idx, m in enumerate(metrics_list):
            val = float(r.hget(name=r_key, key=m))
            anomalies.append(val)
            unified_anomaly = unified_weights[idx] * val
        unified_anomaly = unified_anomaly / sum(unified_weights)
    else:
        _LOGGER.debug("%s - Generating unified anomaly, using max strategy", payload.uuid)
        for idx, m in enumerate(metrics_list):
            val = float(r.hget(name=r_key, key=m))
            anomalies.append(val)
            if unified_anomaly < val:
                unified_anomaly = val

    r.delete(r_key)
    return unified_anomaly, anomalies


def __construct_publisher_payload(
    stream_payload: StreamPayload, final_score: float
) -> PrometheusPayload:
    metric_name = stream_payload.composite_keys["name"]
    namespace = stream_payload.composite_keys["namespace"]

    labels = {
        "model_version": str(stream_payload.get_metadata("version")),
        "namespace": stream_payload.get_metadata("src_labels").get("namespace"),
    }
    subsystem = stream_payload.get_metadata("src_labels").get("hash_id") or None

    return PrometheusPayload(
        timestamp_ms=int(stream_payload.end_ts),
        name=f"{metric_name}_anomaly",
        namespace=namespace,
        subsystem=subsystem,
        type="Gauge",
        value=float(final_score),
        labels=labels,
    )


def __construct_unified_payload(
    stream_payload: StreamPayload, max_anomaly: float, unified_config: UnifiedConf
) -> PrometheusPayload:
    namespace = stream_payload.composite_keys["namespace"]

    labels = {
        "model_version": str(stream_payload.get_metadata("version")),
        "namespace": stream_payload.get_metadata("src_labels").get("namespace"),
        "hash_id": stream_payload.get_metadata("src_labels").get("hash_id"),
    }

    subsystem = stream_payload.get_metadata("src_labels").get("hash_id") or None

    return PrometheusPayload(
        timestamp_ms=int(stream_payload.end_ts),
        name=unified_config.unified_metric_name,
        namespace=namespace,
        subsystem=subsystem,
        type="Gauge",
        value=max_anomaly,
        labels=labels,
    )


def _publish(final_score: float, payload: StreamPayload) -> List[bytes]:
    unified_config = get_unified_config(
        metric=payload.composite_keys["name"], namespace=payload.composite_keys["namespace"]
    )

    publisher_json = __construct_publisher_payload(payload, final_score).as_json()
    _LOGGER.info("%s - Payload sent to publisher: %s", payload.uuid, publisher_json)

    if not unified_config:
        _LOGGER.debug(
            "%s - Using default config, cannot generate a unified anomaly score", payload.uuid
        )
        return [publisher_json]

    try:
        unified_anomaly, anomalies = save_to_redis(
            payload=payload, final_score=final_score, recreate=False, unified_config=unified_config
        )
    except RedisConnectionError:
        _LOGGER.warning("%s - Redis connection failed, recreating the redis client", payload.uuid)
        unified_anomaly, anomalies = save_to_redis(
            payload=payload, final_score=final_score, recreate=True, unified_config=unified_config
        )

    if unified_anomaly > -1:
        unified_json = __construct_unified_payload(
            payload, unified_anomaly, unified_config
        ).as_json()
        _LOGGER.info(
            "%s - Unified anomaly payload sent to publisher: %s", payload.uuid, unified_json
        )
        return [publisher_json, unified_json]
    return [publisher_json]


@msgs_forward
def postprocess(_: str, datum: Datum) -> List[bytes]:
    _start_time = time.perf_counter()

    _in_msg = datum.value.decode("utf-8")
    payload = StreamPayload(**orjson.loads(_in_msg))

    _LOGGER.debug("%s - Received Payload: %r ", payload.uuid, payload)

    metric_config = get_metric_config(
        metric=payload.composite_keys["name"], namespace=payload.composite_keys["namespace"]
    )

    raw_scores = payload.get_stream_array()
    raw_mean_score = np.mean(raw_scores)

    postproc_factory = PostprocessFactory()
    postproc_clf = postproc_factory.get_instance(metric_config.numalogic_conf.postprocess)
    norm_score = postproc_clf.transform(raw_mean_score)

    payload.set_status(Status.POST_PROCESSED)
    _LOGGER.info("%s - Successfully post-processed; final score: %s", payload.uuid, norm_score)

    _LOGGER.debug(
        "%s - Time taken in postprocess: %.4f sec", payload.uuid, time.perf_counter() - _start_time
    )
    return _publish(norm_score, payload)
