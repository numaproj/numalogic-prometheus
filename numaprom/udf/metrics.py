from prometheus_client import Counter

# Metrics
redis_conn_status_count = Counter("numaprom_redis_conn_status_count", "", ["vertex", "status"])


def increase_redis_conn_status(vertex, status):
    redis_conn_status_count.labels(vertex, status).inc()
