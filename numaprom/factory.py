from typing import Callable

from pynumaflow.function import Messages, Server, MultiProcServer
from pynumaflow.sink import Responses, Sink

from numaprom.udf import Preprocess, postprocess, window, metric_filter, inference, threshold
from numaprom.udsink import train, train_rollout


class HandlerFactory:
    @classmethod
    def get_handler(cls, step: str, *args, **kwargs) -> Callable[..., Messages | Responses]:
        match step:
            case "metric_filter":
                return metric_filter
            case "window":
                return window
            case "preprocess":
                return Preprocess(*args, **kwargs)
            case "inference":
                return inference
            case "postprocess":
                return postprocess
            case "threshold":
                return threshold
            case "train":
                return train
            case "train_rollout":
                return train_rollout
            case _:
                raise ValueError(f"Invalid step provided: {step}")


class ServerFactory:
    _SERVERS = {
        "inference": MultiProcServer,
        "train": Sink,
        "train_rollout": Sink,
    }

    @classmethod
    def get_server(cls, step: str, *args, **kwargs) -> Server | Sink:
        server_cls = cls._SERVERS.get(step, Server)
        return server_cls(*args, **kwargs)
