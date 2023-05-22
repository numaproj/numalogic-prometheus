import os
from unittest.mock import patch

import fakeredis
from numalogic.config import NumalogicConf
from omegaconf import OmegaConf

from numaprom import DataConf
from numaprom._config import PipelineConf
from numaprom._constants import TESTS_RESOURCES, DEFAULT_CONFIG_DIR
from numaprom.watcher import ConfigManager

server = fakeredis.FakeServer()
redis_client = fakeredis.FakeStrictRedis(server=server, decode_responses=False)


def mock_configs():
    schema: DataConf = OmegaConf.structured(DataConf)

    conf = OmegaConf.load(os.path.join(TESTS_RESOURCES, "configs", "config.yaml"))
    app_configs = OmegaConf.merge(schema, conf).configs

    conf = OmegaConf.load(os.path.join(TESTS_RESOURCES, "configs", "default-config.yaml"))
    default_configs = OmegaConf.merge(schema, conf).configs

    conf = OmegaConf.load(os.path.join(TESTS_RESOURCES, "configs", "numalogic_config.yaml"))
    schema: NumalogicConf = OmegaConf.structured(NumalogicConf)
    default_numalogic = OmegaConf.merge(schema, conf)

    conf = OmegaConf.load(os.path.join(DEFAULT_CONFIG_DIR, "pipeline_config.yaml"))
    schema: PipelineConf = OmegaConf.structured(PipelineConf)
    pipeline_config = OmegaConf.merge(schema, conf)

    return app_configs, default_configs, default_numalogic, pipeline_config


with patch("numaprom.clients.sentinel.get_redis_client") as mock_get_redis_client:
    with patch(
        "numaprom.clients.sentinel.get_redis_client_from_conf"
    ) as mock_get_redis_client_from_conf:
        mock_get_redis_client_from_conf.return_value = redis_client
        mock_get_redis_client.return_value = redis_client
        with patch.object(ConfigManager, "load_configs") as mock_confs:
            mock_confs.return_value = mock_configs()
            from numaprom.udf import window, preprocess, inference, threshold
            from numaprom.udsink import train, train_rollout


__all__ = [
    "redis_client",
    "window",
    "train",
    "train_rollout",
    "preprocess",
    "inference",
    "threshold",
    "mock_configs",
]
