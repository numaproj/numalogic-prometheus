import os
import sys
import json
import torch
import datetime
from unittest.mock import MagicMock
from mlflow.entities.model_registry import ModelVersion

from pynumaflow.function import Datum
from numalogic.models.autoencoder.variants import VanillaAE

from nlogicprom.constants import TESTS_DIR
from nlogicprom.factory import HandlerFactory
from nlogicprom.tests import *

sys.modules["nlogicprom.mlflow"] = MagicMock()
MODEL_DIR = os.path.join(TESTS_DIR, "resources", "models")


def get_datum(data: str or bytes) -> Datum:
    if type(data) is not bytes:
        data = json.dumps(data).encode("utf-8")

    return Datum(value=data, event_time=datetime.datetime.now(), watermark=datetime.datetime.now())


def get_prepoc_input(data_path: str) -> Datum:
    with open(data_path) as fp:
        data = json.load(fp)
    output_ = None
    for obj in data:
        output_ = window(None, get_datum(obj))
    return get_datum(output_.items()[0].value) if output_ else None


def get_inference_input(data_path: str) -> Datum:
    preproc_input = get_prepoc_input(data_path)
    handler_ = HandlerFactory.get_handler("preprocess")
    out = handler_(None, preproc_input)
    return get_datum(out.items()[0].value)


def get_postproc_input(data_path: str) -> Datum:
    inference_input = get_inference_input(data_path)
    handler_ = HandlerFactory.get_handler("inference")
    out = handler_(None, inference_input)
    return get_datum(out.items()[0].value)


def return_mock_cpu_load(*_, **__):
    return {
        "primary_artifact": VanillaAE(12),
        "metadata": torch.load(os.path.join(MODEL_DIR, "model_cpu.pth")),
        "model_properties": ModelVersion(
            creation_timestamp=1656615600000,
            current_stage="Production",
            description="",
            last_updated_timestamp=datetime.datetime.now().timestamp() * 1000,
            name="sandbox:lol::demo:lol",
            run_id="6f1e582fb6194bbdaa4141feb2ce8e27",
            run_link="",
            source="mlflow-artifacts:/0/6f1e582fb6194bbdaa4141feb2ce8e27/artifacts/model",
            status="READY",
            status_message="",
            tags={},
            user_id="",
            version="125",
        ),
    }
