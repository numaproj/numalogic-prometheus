from typing import Callable

from pynumaflow.function import Messages

from nlogicprom.udf import preprocess, postprocess, window, metric_filter, inference


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

        raise NotImplementedError(f"Invalid step provided: {step}")
