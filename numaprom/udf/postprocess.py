import os
import time
from typing import List

import numpy as np
from numalogic.models.threshold import StaticThreshold
from numalogic.postprocess import TanhNorm
from orjson import orjson
from pynumaflow.function import Datum
from redis.exceptions import ConnectionError as RedisConnectionError

from numaprom import get_logger
from numaprom.config._config import UnifiedConf
from numaprom.entities import Status, PrometheusPayload, StreamPayload, Header
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
    max_anomaly = -1
    anomalies = []
    for m in metrics_list:
        val = float(r.hget(name=r_key, key=m))
        anomalies.append(val)
        if max_anomaly < val:
            max_anomaly = val
    r.delete(r_key)

    return max_anomaly, anomalies


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
        max_anomaly, anomalies = save_to_redis(
            payload=payload, final_score=final_score, recreate=False, unified_config=unified_config
        )
    except RedisConnectionError:
        _LOGGER.warning("%s - Redis connection failed, recreating the redis client", payload.uuid)
        max_anomaly, anomalies = save_to_redis(
            payload=payload, final_score=final_score, recreate=True, unified_config=unified_config
        )

    if max_anomaly > -1:
        unified_json = __construct_unified_payload(payload, max_anomaly, unified_config).as_json()
        _LOGGER.info(
            "%s - Unified anomaly payload sent to publisher: %s", payload.uuid, unified_json
        )
        return [publisher_json, unified_json]
    return [publisher_json]


def calculate_ensemble_score(payload: StreamPayload) -> float:
    # Load config
    metric_config = get_metric_config(
        metric=payload.composite_keys["name"], namespace=payload.composite_keys["namespace"]
    )

    # Get static scores
    static_clf = StaticThreshold(upper_limit=metric_config.static_threshold)
    static_scores = static_clf.score_samples(payload.get_stream_array())
    mean_static_score = np.mean(static_scores)

    # Get model scores
    model_scores = payload.get_stream_array()
    mean_model_score = np.mean(model_scores)

    # Perform score normalization
    postproc_clf = TanhNorm()
    norm_static_score = postproc_clf.transform(mean_static_score)
    norm_model_score = postproc_clf.transform(mean_model_score)

    # Calculate ensemble score
    static_wt = metric_config.static_threshold_wt
    model_wt = 1.0 - metric_config.static_threshold_wt
    ensemble_score = (static_wt * norm_static_score) + (model_wt * norm_model_score)

    _LOGGER.debug(
        "%s - Model score: %s, Static score: %s, Static wt: %s",
        payload.uuid,
        norm_model_score,
        norm_static_score,
        static_wt,
    )

    return ensemble_score


@msgs_forward
def postprocess(_: str, datum: Datum) -> List[bytes]:
    _start_time = time.perf_counter()

    _in_msg = datum.value.decode("utf-8")
    payload = StreamPayload(**orjson.loads(_in_msg))

    _LOGGER.debug("%s - Received Payload: %r ", payload.uuid, payload)

    print(payload.header)
    # Use only using static thresholding
    if payload.header == Header.STATIC_INFERENCE:
        raw_scores = payload.get_stream_array()
        raw_mean_score = np.mean(raw_scores)

        postproc_clf = TanhNorm()
        final_score = postproc_clf.transform(raw_mean_score)
        _LOGGER.info("%s - Final static threshold score: %s", payload.uuid, final_score)

    # Compute ensemble score if model score exists
    else:
        final_score = calculate_ensemble_score(payload)
        _LOGGER.info("%s - Final ensemble score: %s", payload.uuid, final_score)

    payload.set_status(Status.POST_PROCESSED)
    _LOGGER.debug(
        "%s - Time taken in postprocess: %.4f sec", payload.uuid, time.perf_counter() - _start_time
    )
    return _publish(final_score, payload)
