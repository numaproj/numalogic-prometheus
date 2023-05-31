import logging
import sys
import threading

from numaprom._constants import CONFIG_PATHS
from numaprom.factory import HandlerFactory, ServerFactory
from numaprom.watcher import Watcher, ConfigHandler

LOGGER = logging.getLogger(__name__)


def run_watcher():
    w = Watcher(CONFIG_PATHS, ConfigHandler())
    w.run()


if __name__ == "__main__":
    background_thread = threading.Thread(target=run_watcher, args=())
    background_thread.daemon = True
    background_thread.start()

    step = sys.argv[1]
    step_function = HandlerFactory.get_handler(step)
    step_server = ServerFactory.get_server(step, step_function, sock_path="/tmp/numaprom.sock")

    LOGGER.info("Starting %s server for step: %s", step_server.__class__.__name__, step)
    step_server.start()
