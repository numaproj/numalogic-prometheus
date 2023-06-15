from numaprom.udf import window, keying, aggregate


class HandlerFactory:
    @classmethod
    def get_handler(cls, step: str):

        if step == "keying":
            return keying

        if step == "aggregate":
            return aggregate

        if step == "window":
            return window

        raise NotImplementedError(f"Invalid step provided: {step}")
