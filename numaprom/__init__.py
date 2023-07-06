import logging
import os
import sys

from loguru import logger
from numaprom._config import UnifiedConf, MetricConf, AppConf, DataConf


class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def __get_logger() -> logger:
    # Collect logs from logging library
    logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)
    logging.basicConfig(handlers=[InterceptHandler()], level=0)
    logger.remove()

    # define log sink
    sink = sys.stderr

    # define library log levels
    filter_levels = {
        "watchdog": "WARNING",
        "pytorch_lightning": "WARNING",
        "pynumaflow": "WARNING",
    }
    formatter = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> "
        "| <level>{message}</level>"
    )
    if os.getenv("DEBUG", False):
        filter_levels["numalogic"] = "DEBUG"
        filter_levels["numaprom"] = "DEBUG"
        logger.add(
            sink=sink,
            level="DEBUG",
            colorize=True,
            filter=filter_levels,
            format=formatter,
            diagnose=False,
        )
    else:
        filter_levels["numalogic"] = "INFO"
        filter_levels["numaprom"] = "INFO"
        logger.add(
            sink=sink,
            level="INFO",
            colorize=True,
            filter=filter_levels,
            format=formatter,
            diagnose=False,
        )
    logger.info("Starting Logger...")
    return logger


# get Logger
LOGGER = __get_logger()

__all__ = ["UnifiedConf", "MetricConf", "AppConf", "DataConf", "LOGGER"]
