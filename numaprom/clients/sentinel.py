import os

from numalogic.tools.types import redis_client_t
from redis.backoff import ExponentialBackoff
from redis.exceptions import RedisClusterException, RedisError
from redis.retry import Retry
from redis.sentinel import Sentinel, MasterNotFoundError

from numaprom import LOGGER
from numaprom._config import RedisConf
from numaprom.watcher import ConfigManager


SENTINEL_CLIENT: redis_client_t | None = None


def get_redis_client(
    host: str,
    port: int,
    password: str,
    mastername: str,
    recreate: bool = False,
    master_node: bool = True,
    reset: bool = False,
) -> redis_client_t:
    """Return a master redis client for sentinel connections, with retry.

    Args:
    ----
        host: Redis host
        port: Redis port
        password: Redis password
        mastername: Redis sentinel master name
        decode_responses: Whether to decode responses
        recreate: Whether to flush and recreate the client
        master_node: Whether to use the master node or the slave nodes

    Returns
    -------
        Redis client instance
    """
    global SENTINEL_CLIENT

    if reset:
        LOGGER.info("Reset Sentinel Client to None")
        SENTINEL_CLIENT = None

    if not recreate and SENTINEL_CLIENT:
        LOGGER.info("Reusing Existing Sentinel Client")
        return SENTINEL_CLIENT

    retry = Retry(
        ExponentialBackoff(),
        3,
        supported_errors=(
            ConnectionError,
            TimeoutError,
            RedisClusterException,
            RedisError,
            MasterNotFoundError,
        ),
    )

    conn_kwargs = {
        "socket_timeout": 1,
        "socket_connect_timeout": 1,
        "socket_keepalive": True,
        "health_check_interval": 10,
    }

    sentinel = Sentinel(
        [(host, port)],
        sentinel_kwargs=dict(password=password, **conn_kwargs),
        retry=retry,
        password=password,
        **conn_kwargs
    )
    if master_node:
        LOGGER.info("Creating Master Sentinel Redis Client")
        SENTINEL_CLIENT = sentinel.master_for(mastername)
    else:
        LOGGER.info("Creating Slave Sentinel Redis Client")
        SENTINEL_CLIENT = sentinel.slave_for(mastername)
    LOGGER.info(
        "Sentinel redis params: {args}, master_node: {is_master}",
        args=conn_kwargs,
        is_master=master_node,
    )

    return SENTINEL_CLIENT


def get_redis_client_from_conf(redis_conf: RedisConf = None, **kwargs) -> redis_client_t:
    """Return a master redis client from config for sentinel connections, with retry.

    Args:
    ----
        redis_conf: RedisConf object with host, port, master_name, etc.
        **kwargs: Additional arguments to pass to get_redis_client.

    Returns
    -------
        Redis client instance
    """
    if not redis_conf:
        redis_conf = ConfigManager.get_redis_config()

    return get_redis_client(
        redis_conf.host,
        redis_conf.port,
        password=os.getenv("REDIS_AUTH"),
        mastername=redis_conf.master_name,
        **kwargs
    )
