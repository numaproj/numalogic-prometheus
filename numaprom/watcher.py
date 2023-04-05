import os
import time
from typing import Optional

from omegaconf import OmegaConf
from watchdog.observers import Observer
from numalogic.config import NumalogicConf
from watchdog.events import FileSystemEventHandler

from numaprom.redis import get_redis_client
from numaprom._constants import CONFIG_DIR, DEFAULT_CONFIG_DIR
from numaprom import NumapromConf, get_logger, ServiceConf, UnifiedConf, MetricConf

_LOGGER = get_logger(__name__)

HOST = os.getenv("REDIS_HOST")
PORT = os.getenv("REDIS_PORT")
AUTH = os.getenv("REDIS_AUTH")


def get_app_config(namespace: str, metric: str):
    r = get_redis_client(HOST, PORT, password=AUTH)

    # search and load from app configs
    app_config = r.hgetall(f"{namespace}")

    # if not search and load from default configs
    if not app_config:
        for key, _conf in r.hgetall("default-configs").items():
            if metric in _conf.unified_configs[0].unified_metrics:
                app_config = [_conf]
                break

    # if not in default configs, initialize Namespace conf with default values
    if not app_config:
        app_config = OmegaConf.structured(ServiceConf)
    else:
        app_config = app_config[0]

    # loading and setting default numalogic config
    for metric_config in app_config.metric_configs:
        if OmegaConf.is_missing(metric_config, "numalogic_conf"):
            metric_config.numalogic_conf = r.hgetall("default-numalogic")

    return app_config


def get_metric_config(namespace: str, metric: str) -> Optional[MetricConf]:
    app_config = get_app_config(namespace, metric)
    metric_config = list(
        filter(lambda conf: (conf.metric == metric), app_config.metric_configs)
    )
    if not metric_config:
        return app_config.metric_configs[0]
    return metric_config[0]


def get_unified_config(namespace: str, metric: str) -> Optional[UnifiedConf]:
    app_config = get_app_config(namespace, metric)
    unified_config = list(
        filter(lambda conf: (metric in conf.unified_metrics), app_config.unified_configs)
    )
    if not unified_config:
        return None
    return unified_config[0]


class ConfigHandler(FileSystemEventHandler):
    def __init__(self):
        self.redis_client = get_redis_client(HOST, PORT, password=AUTH)
        self.np_schema: NumapromConf = OmegaConf.structured(NumapromConf)
        self.nl_schema: NumalogicConf = OmegaConf.structured(NumalogicConf)

    def update_app_configs(self):
        _conf = OmegaConf.load(os.path.join(CONFIG_DIR, "config.yaml"))
        _app_configs = OmegaConf.merge(self.np_schema, _conf).configs

        for _config in _app_configs:
            self.redis_client.hset(f"{_config.namespace}", mapping=_config)

        _LOGGER.info("Successfully updated app configs - %s", _app_configs)

    def update_default_configs(self):
        _conf = OmegaConf.load(os.path.join(DEFAULT_CONFIG_DIR, "config.yaml"))
        _default_configs = OmegaConf.merge(self.np_schema, _conf).configs

        _mapping = dict(map(lambda config: (config.namespace, config), _default_configs))
        self.redis_client.hset("default-configs", mapping=_mapping)

        _LOGGER.info("Successfully updated default configs - %s", _default_configs)

    def update_default_numalogic(self):
        _conf = OmegaConf.load(os.path.join(DEFAULT_CONFIG_DIR, "numalogic_config.yaml"))
        _default_numalogic = OmegaConf.merge(self.nl_schema, _conf)

        self.redis_client.hset("default-numalogic", mapping=_default_numalogic)
        _LOGGER.info("Successfully updated default numalogic config - %s", _default_numalogic)

    def on_any_event(self, event):

        if event.event_type == 'created' or event.event_type == 'modified':
            _file = os.path.basename(event.src_path)
            _dir = os.path.basename(os.path.dirname(event.src_path))

            _LOGGER.info("Watchdog received %s event - %s/%s", event.event_type, _dir, _file)
            if _dir == 'configs' and _file == 'config.yaml':
                self.update_app_configs()

            if _dir == 'default-configs' and _file == 'config.yaml':
                self.update_default_configs()

            if _dir == 'default-configs' and _file == 'numalogic_config.yaml':
                self.update_default_numalogic()


class Watcher:
    def __init__(self, directories=None, handler=FileSystemEventHandler()):
        if directories is None:
            directories = ["."]
        self.observer = Observer()
        self.handler = handler
        self.directories = directories

    def run(self):
        for directory in self.directories:
            self.observer.schedule(self.handler, directory, recursive=True)
            _LOGGER.info("\nWatcher Running in {}/\n".format(directory))

        self.observer.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.observer.stop()
        self.observer.join()
        _LOGGER.info("\nWatcher Terminated\n")
