import logging
from unittest.mock import patch

import fakeredis

server = fakeredis.FakeServer()
redis_client = fakeredis.FakeStrictRedis(server=server, decode_responses=True)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)


formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
stream_handler.setFormatter(formatter)

LOGGER.addHandler(stream_handler)


with patch("numaprom.redis.get_redis_client") as mock_get_redis_client:
    mock_get_redis_client.return_value = redis_client
    from numaprom.udf import window
    from numaprom.udsink import train, train_rollout


__all__ = ["redis_client", "window", "train", "train_rollout"]
