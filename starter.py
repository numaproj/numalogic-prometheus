import sys

from pynumaflow.function import UserDefinedFunctionServicer
from pynumaflow.sink import UserDefinedSinkServicer

from numaprom._constants import CONFIG_PATHS
from numaprom.factory import HandlerFactory
from numaprom.watcher import Watcher, ConfigHandler

if __name__ == "__main__":
    step_handler = HandlerFactory.get_handler(sys.argv[2])
    server_type = sys.argv[1]

    if server_type == "udsink":
        server = UserDefinedSinkServicer(step_handler)
    elif server_type == "udf":
        server = UserDefinedFunctionServicer(step_handler)
    else:
        raise ValueError(f"sys arg: {server_type} not understood!")

    server.start()

    w = Watcher(CONFIG_PATHS, ConfigHandler())
    w.run()
