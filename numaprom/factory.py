from typing import Callable, Union

from pynumaflow.function import Messages
from pynumaflow.sink import Responses

from numaprom.udf import preprocess, postprocess, window, metric_filter, inference
from numaprom.udsink import train, train_rollout


class HandlerFactory:
    @classmethod
    def get_handler(cls, step: str) -> Callable[..., Union[Messages, Responses]]:
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

        if step == "train_rollout":
            return train_rollout

        raise NotImplementedError(f"Invalid step provided: {step}")
