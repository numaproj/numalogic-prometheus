import sys
import aiorun

from pynumaflow.function import Server, AsyncServer
from pynumaflow.sink import Sink

from numaprom.factory import HandlerFactory

if __name__ == "__main__":
    step_handler = HandlerFactory.get_handler(sys.argv[2])
    server_type = sys.argv[1]

    if server_type == "udsink":
        server = Sink(sink_handler=step_handler)
        server.start()
    elif server_type == "udf":
        server = Server(map_handler=step_handler)
        server.start()
    elif server_type == "async_udf":
        server = AsyncServer(reduce_handler=step_handler)
        aiorun.run(server.start())
    else:
        raise ValueError(f"sys arg: {server_type} not understood!")
