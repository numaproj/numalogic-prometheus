import sys

from pynumaflow.function import UserDefinedFunctionServicer

from nlogicprom.factory import HandlerFactory

if __name__ == "__main__":
    step_handler = HandlerFactory.get_handler(sys.argv[1])
    handler = UserDefinedFunctionServicer(step_handler)
    handler.start()
