import sys
import threading

from pynumaflow.function import Server
from pynumaflow.sink import Sink
from numaprom.metrics import metrics

from numaprom._constants import CONFIG_PATHS
from numaprom.factory import HandlerFactory
from numaprom.watcher import Watcher, ConfigHandler


def run_watcher():
    w = Watcher(CONFIG_PATHS, ConfigHandler())
    w.run()


if __name__ == "__main__":
    background_thread = threading.Thread(target=run_watcher, args=())
    background_thread.daemon = True
    background_thread.start()

    step_handler = HandlerFactory.get_handler(sys.argv[2])
    metrics.start_metrics_server(8490)
    server_type = sys.argv[1]

    if server_type == "udsink":
        server = Sink(step_handler)
    elif server_type == "udf":
        server = Server(step_handler)
    else:
        raise ValueError(f"sys arg: {server_type} not understood!")

    server.start()
