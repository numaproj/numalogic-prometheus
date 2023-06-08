import logging
import os

from numaprom._config import UnifiedConf, MetricConf, AppConf, DataConf


def get_logger(name):
    formatter = logging.Formatter("%(asctime)s-%(levelname)s-%(message)s")
    logger = logging.getLogger(name)

    numalogic_logger = logging.getLogger("numalogic")
    pl_logger = logging.getLogger("pytorch_lightning")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)

    if os.getenv("DEBUG", False):
        logger.setLevel(logging.DEBUG)
        stream_handler.setLevel(logging.DEBUG)
        numalogic_logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
        stream_handler.setLevel(logging.INFO)
        numalogic_logger.setLevel(logging.INFO)

    pl_logger.setLevel(logging.ERROR)
    numalogic_logger.propagate = True
    pl_logger.propagate = False
    numalogic_logger.addHandler(stream_handler)
    pl_logger.addHandler(stream_handler)
    return logger


__all__ = ["UnifiedConf", "MetricConf", "AppConf", "DataConf", "get_logger"]
