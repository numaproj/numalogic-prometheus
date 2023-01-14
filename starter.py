import sys

from pynumaflow.function import UserDefinedFunctionServicer
from pynumaflow.sink import UserDefinedSinkServicer

from numaprom.factory import HandlerFactory

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
