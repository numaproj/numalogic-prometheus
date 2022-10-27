import json
import logging
from redis.cluster import RedisCluster

from nlogicprom.tools import is_host_reachable

_LOGGER = logging.getLogger(__name__)
redis_client: RedisCluster = None


def get_redis_client(
    host: str, port: str, password: str = None, decode_responses: bool = True, recreate=False
) -> RedisCluster:
    global redis_client

    if not recreate and redis_client:
        return redis_client

    redis_params = {
        "host": host,
        "port": port,
        "password": password,
        "decode_responses": decode_responses,
        "skip_full_coverage_check": True,
        "dynamic_startup_nodes": False,
    }
    _LOGGER.info("Redis params: %s", json.dumps(redis_params, indent=4))

    if not is_host_reachable(host, port):
        _LOGGER.error("Redis Cluster is unreachable after retries!")

    redis_client = RedisCluster(**redis_params)
    return redis_client
