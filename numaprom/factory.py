from typing import Callable

from pynumaflow.function import Messages

from numaprom.udf import preprocess, postprocess, window, metric_filter, inference
from numaprom.udsink import train


class HandlerFactory:
    @classmethod
    def get_handler(cls, step: str) -> Callable[..., Messages]:
        if step == "metric_filter":
            return metric_filter

        if step == "window":
            return window

        if step == "preprocess":
            return preprocess

        if step == "inference":
            return inference

        if step == "postprocess":
            return postprocess

        if step == "train":
            return train

        raise NotImplementedError(f"Invalid step provided: {step}")
