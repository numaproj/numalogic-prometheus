from prometheus_client import Counter

# Metrics
redis_conn_status_count = Counter("numaprom_redis_conn_status_count", "", ["vertex", "status"])


def increase_redis_conn_status(vertex, status):
    redis_conn_status_count.labels(vertex, status).inc()


inference_count = Counter(
    "numaprom_inference_count", "", ["model", "namespace", "app", "metric", "status"]
)


def increase_interface_count(model, namespace, app, metric, status):
    inference_count.labels(model, namespace, app, metric, status).inc()
