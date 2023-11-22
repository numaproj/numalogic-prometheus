from collections.abc import Callable

from pynumaflow.mapper import Messages
from pynumaflow.sinker import Responses

from numaprom.udf import preprocess, postprocess, window, inference, threshold
from numaprom.udsink import train, train_rollout


class HandlerFactory:
    @classmethod
    def get_handler(cls, step: str) -> Callable[..., Messages | Responses]:
        if step == "window":
            return window

        if step == "preprocess":
            return preprocess

        if step == "inference":
            return inference

        if step == "postprocess":
            return postprocess

        if step == "threshold":
            return threshold

        if step == "train":
            return train

        if step == "train_rollout":
            return train_rollout

        raise NotImplementedError(f"Invalid step provided: {step}")
