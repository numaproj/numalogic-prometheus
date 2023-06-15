import os
import time

import numpy as np
from orjson import orjson
from pynumaflow.function import Datum
from redis.exceptions import RedisError, RedisClusterException
from redis.sentinel import MasterNotFoundError

from numaprom import LOGGER, UnifiedConf
from numaprom.clients.sentinel import get_redis_client_from_conf
from numaprom.entities import Status, PrometheusPayload, StreamPayload, Header
from numaprom.tools import msgs_forward, WindowScorer
from numaprom.watcher import ConfigManager

AUTH = os.getenv("REDIS_AUTH")


def __save_to_redis(
    payload: StreamPayload, final_score: float, recreate: bool, unified_config: UnifiedConf
):
    r = get_redis_client_from_conf(recreate=recreate)

    metric_name = payload.composite_keys["name"]

    r_keys = payload.composite_keys
    r_keys.pop("name")
    r_key = f"{':'.join(r_keys.values())}:{payload.end_ts}"

    final_score = -1 if np.isnan(final_score) else final_score
    r.hset(r_key, mapping={metric_name: final_score})
    LOGGER.info(
        "{uuid} - Saved to redis, redis_key: {redis_key}, metric: {metric_name}, "
        "anomaly_score: {final_score}",
        uuid=payload.uuid,
        redis_key=r_key,
        metric_name=metric_name,
        final_score=final_score,
    )

    anomalies = []
    for m in unified_config.unified_metrics:
        if r.hexists(name=r_key, key=m):
            anomalies.append(float(r.hget(name=r_key, key=m).decode()))
        else:
            LOGGER.debug(
                "{uuid} - Unable to generate unified anomaly, missing metric: "
                "{metric}, redis_key: {redis_key}",
                uuid=payload.uuid,
                metric=m,
                redis_key=r_key,
            )
            return -1, []

    LOGGER.debug("{uuid} - Received all metrics, generating unified anomaly", uuid=payload.uuid)
    unified_weights = unified_config.unified_weights
    if unified_weights:
        weighted_anomalies = np.multiply(anomalies, unified_weights)
        unified_anomaly = float(np.sum(weighted_anomalies) / np.sum(unified_weights))
        LOGGER.info(
            "{uuid} - Generating unified anomaly, using unified weights. "
            "Unified Anomaly: {anomaly}",
            uuid=payload.uuid,
            anomaly=unified_anomaly,
        )
    else:
        unified_anomaly = max(anomalies)
        LOGGER.info(
            "{uuid} - Generated unified anomaly, using max strategy. Unified Anomaly: {anomaly}",
            uuid=payload.uuid,
            anomaly=unified_anomaly,
        )

    r.delete(r_key)
    return unified_anomaly, anomalies


def __construct_publisher_payload(
    stream_payload: StreamPayload, final_score: float
) -> PrometheusPayload:
    metric_name = stream_payload.composite_keys["name"]
    namespace = stream_payload.composite_keys["namespace"]

    labels = {"model_version": str(stream_payload.get_metadata("version"))}

    for key in stream_payload.composite_keys:
        if key != "name":
            labels[key] = stream_payload.composite_keys[key]

    return PrometheusPayload(
        timestamp_ms=int(stream_payload.end_ts),
        name=f"{metric_name}_anomaly",
        namespace=namespace,
        subsystem=None,
        type="Gauge",
        value=float(final_score),
        labels=labels,
    )


def __construct_unified_payload(
    stream_payload: StreamPayload, max_anomaly: float, unified_config: UnifiedConf
) -> PrometheusPayload:
    namespace = stream_payload.composite_keys["namespace"]

    labels = {"model_version": str(stream_payload.get_metadata("version"))}

    for key in stream_payload.composite_keys:
        if key != "name":
            labels[key] = stream_payload.composite_keys[key]

    return PrometheusPayload(
        timestamp_ms=int(stream_payload.end_ts),
        name=unified_config.unified_metric_name,
        namespace=namespace,
        subsystem=None,
        type="Gauge",
        value=max_anomaly,
        labels=labels,
    )


def _publish(final_score: float, payload: StreamPayload) -> list[bytes]:
    unified_config = ConfigManager.get_unified_config(payload.composite_keys)

    publisher_json = __construct_publisher_payload(payload, final_score).as_json()
    LOGGER.info(
        "{uuid} - Payload sent to publisher: {publisher_json}",
        uuid=payload.uuid,
        publisher_json=publisher_json,
    )

    if not unified_config:
        LOGGER.debug(
            "{uuid} - Using default config, cannot generate a unified anomaly score",
            uuid=payload.uuid,
        )
        return [publisher_json]

    try:
        unified_anomaly, anomalies = __save_to_redis(
            payload=payload, final_score=final_score, recreate=False, unified_config=unified_config
        )
    except (RedisError, RedisClusterException, MasterNotFoundError) as warn:
        LOGGER.warning(
            "{uuid} - Redis connection failed, recreating the redis client; err: {warn}",
            uuid=payload.uuid,
            warn=warn,
        )
        unified_anomaly, anomalies = __save_to_redis(
            payload=payload, final_score=final_score, recreate=True, unified_config=unified_config
        )

    if unified_anomaly > -1:
        unified_json = __construct_unified_payload(
            payload, unified_anomaly, unified_config
        ).as_json()
        LOGGER.info(
            "{uuid} - Unified anomaly payload sent to publisher: {unified_json}",
            uuid=payload.uuid,
            unified_json=unified_json,
        )
        return [publisher_json, unified_json]
    return [publisher_json]


@msgs_forward
def postprocess(_: list[str], datum: Datum) -> list[bytes]:
    """UDF for performing the following steps:

    1. Postprocess the raw scores, e.g. bring the scores into a range of 0 - 10
    2. Calculate a unified anomaly score by combining multiple metrics
    3. Construct and publish a Prometheus Payload object
    """
    _start_time = time.perf_counter()

    _in_msg = datum.value.decode("utf-8")
    payload = StreamPayload(**orjson.loads(_in_msg))

    # Load config
    metric_config = ConfigManager.get_metric_config(payload.composite_keys)

    LOGGER.debug("{uuid} - Received Payload: {payload} ", uuid=payload.uuid, payload=payload)

    winscorer = WindowScorer(metric_config)

    # Use only using static thresholding
    if payload.header == Header.STATIC_INFERENCE:
        final_score = winscorer.get_winscore(payload)
        LOGGER.info(
            "{uuid} - Final static threshold score: {final_score}, keys: {keys}",
            uuid=payload.uuid,
            final_score=final_score,
            keys=payload.composite_keys,
        )

    # Compute ensemble score otherwise
    else:
        final_score = winscorer.get_final_winscore(payload)
        LOGGER.info(
            "{uuid} - Final ensemble score: {ensemble_score}, "
            "static thresh wt: {thresh}, keys: {keys}",
            uuid=payload.uuid,
            ensemble_score=final_score,
            thresh=metric_config.static_threshold_wt,
            keys=payload.composite_keys,
        )

    payload.set_status(Status.POST_PROCESSED)
    messages = _publish(final_score, payload)
    LOGGER.debug(
        "{uuid} - Time taken in postprocess: {time} sec",
        uuid=payload.uuid,
        time=time.perf_counter() - _start_time,
    )
    return messages
