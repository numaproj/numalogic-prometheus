import os
import time
from functools import cache
from typing import Optional

from omegaconf import OmegaConf
from watchdog.observers import Observer
from numalogic.config import NumalogicConf
from watchdog.events import FileSystemEventHandler

from numaprom._constants import CONFIG_DIR, DEFAULT_CONFIG_DIR
from numaprom import NumapromConf, get_logger, AppConf, MetricConf, UnifiedConf

_LOGGER = get_logger(__name__)

CONFIG = {}


class ConfigManager:
    @staticmethod
    def load_configs():
        schema: NumapromConf = OmegaConf.structured(NumapromConf)

        conf = OmegaConf.load(os.path.join(CONFIG_DIR, "config.yaml"))
        app_configs = OmegaConf.merge(schema, conf).configs

        conf = OmegaConf.load(os.path.join(DEFAULT_CONFIG_DIR, "config.yaml"))
        default_configs = OmegaConf.merge(schema, conf).configs

        conf = OmegaConf.load(os.path.join(DEFAULT_CONFIG_DIR, "numalogic_config.yaml"))
        schema: NumalogicConf = OmegaConf.structured(NumalogicConf)
        default_numalogic = OmegaConf.merge(schema, conf)

        return app_configs, default_configs, default_numalogic

    def update_configs(self):
        app_configs, default_configs, default_numalogic = self.load_configs()

        CONFIG["app_configs"] = dict()
        for _config in app_configs:
            CONFIG["app_configs"][_config.namespace] = _config

        CONFIG["default_configs"] = dict(map(lambda c: (c.namespace, c), default_configs))
        CONFIG["default_numalogic"] = default_numalogic

        _LOGGER.info("Successfully updated configs - %s", CONFIG)
        return CONFIG

    @cache
    def get_app_config(self, metric: str, namespace: str) -> Optional[AppConf]:
        if not CONFIG:
            self.update_configs()

        app_config = None

        # search and load from app configs
        if namespace in CONFIG["app_configs"]:
            app_config = CONFIG["app_configs"][namespace]

        # if not search and load from default configs
        if not app_config:
            for key, _conf in CONFIG["default_configs"].items():
                if metric in _conf.unified_configs[0].unified_metrics:
                    app_config = _conf
                    break

        # if not in default configs, initialize Namespace conf with default values
        if not app_config:
            app_config = OmegaConf.structured(AppConf)

        # loading and setting default numalogic config
        for metric_config in app_config.metric_configs:
            if OmegaConf.is_missing(metric_config, "numalogic_conf"):
                metric_config.numalogic_conf = CONFIG["default_numalogic"]

        return app_config

    def get_metric_config(self, composite_keys: dict) -> Optional[MetricConf]:
        app_config = self.get_app_config(
            metric=composite_keys["name"], namespace=composite_keys["namespace"]
        )
        metric_config = list(
            filter(lambda conf: (conf.metric == composite_keys["name"]), app_config.metric_configs)
        )
        if not metric_config:
            return app_config.metric_configs[0]
        return metric_config[0]

    def get_unified_config(self, composite_keys: dict) -> Optional[UnifiedConf]:
        app_config = self.get_app_config(
            metric=composite_keys["name"], namespace=composite_keys["namespace"]
        )
        unified_config = list(
            filter(
                lambda conf: (composite_keys["name"] in conf.unified_metrics),
                app_config.unified_configs,
            )
        )
        if not unified_config:
            return None
        return unified_config[0]


class ConfigHandler(FileSystemEventHandler):
    def ___init__(self):
        self.config_manger = ConfigManager()

    def on_any_event(self, event):
        if event.event_type == "created" or event.event_type == "modified":
            _file = os.path.basename(event.src_path)
            _dir = os.path.basename(os.path.dirname(event.src_path))

            _LOGGER.info("Watchdog received %s event - %s/%s", event.event_type, _dir, _file)
            self.config_manger.update_configs()


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