import logging
import os

LOGGER = logging.getLogger(__name__)

stream_handler = logging.StreamHandler()

if os.getenv("DEBUG", False):
    LOGGER.setLevel(logging.DEBUG)
    stream_handler.setLevel(logging.DEBUG)
else:
    LOGGER.setLevel(logging.INFO)
    stream_handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
stream_handler.setFormatter(formatter)


LOGGER.addHandler(stream_handler)
pl_logger = logging.getLogger("pytorch_lightning")
pl_logger.propagate = False
pl_logger.setLevel(logging.ERROR)
