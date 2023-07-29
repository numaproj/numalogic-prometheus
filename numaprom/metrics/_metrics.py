from functools import partial

from prometheus_client import start_http_server
from prometheus_client import Counter

from numaprom import LOGGER

# Metrics
REDIS_CONN_STATUS_COUNT = Counter("numaprom_redis_conn_status_count", "", ["vertex", "status"])
INFERENCE_COUNT = Counter(
    "numaprom_inference_count", "", ["model", "namespace", "app", "metric", "status"]
)


def increase_redis_conn_status(vertex: str, status: str) -> None:
    global REDIS_CONN_STATUS_COUNT
    REDIS_CONN_STATUS_COUNT.labels(vertex, status).inc()


inc_redis_conn_success = partial(increase_redis_conn_status, status="success")
inc_redis_conn_failed = partial(increase_redis_conn_status, status="failed")


def inc_inference_count(model: str, namespace: str, app: str, metric: str, status: str) -> None:
    global INFERENCE_COUNT
    INFERENCE_COUNT.labels(model, namespace, app, metric, status).inc()


def start_metrics_server(port: int) -> None:
    LOGGER.info("Starting Prometheus metrics server on port: {port}", port=port)
    start_http_server(port)
