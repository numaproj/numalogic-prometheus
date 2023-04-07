import os
import time

from omegaconf import OmegaConf
from watchdog.observers import Observer
from numalogic.config import NumalogicConf
from watchdog.events import FileSystemEventHandler

from numaprom._constants import CONFIG_DIR, DEFAULT_CONFIG_DIR
from numaprom import NumapromConf, get_logger

_LOGGER = get_logger(__name__)

config = {}


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


def update_configs():
    app_configs, default_configs, default_numalogic = load_configs()

    config["app_configs"] = dict()
    for _config in app_configs:
        config["app_configs"][_config.namespace] = _config

    config["default_configs"] = dict(map(lambda c: (c.namespace, c), default_configs))
    config["default_numalogic"] = default_numalogic

    _LOGGER.info("Successfully updated configs - %s", config)
    return config


class ConfigHandler(FileSystemEventHandler):
    def on_any_event(self, event):
        if event.event_type == 'created' or event.event_type == 'modified':
            _file = os.path.basename(event.src_path)
            _dir = os.path.basename(os.path.dirname(event.src_path))

            _LOGGER.info("Watchdog received %s event - %s/%s", event.event_type, _dir, _file)
            update_configs()


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
