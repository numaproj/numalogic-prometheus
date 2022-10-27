from unittest.mock import patch

import fakeredis

server = fakeredis.FakeServer()
redis_client = fakeredis.FakeStrictRedis(server=server, decode_responses=True)


with patch("nlogicprom.redis.get_redis_client") as mock_get_redis_client:
    mock_get_redis_client.return_value = redis_client
    from nlogicprom.udf import window


__all__ = ["redis_client", "window"]
