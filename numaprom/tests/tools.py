import os
import sys
import json
from unittest import mock

import torch
import datetime
from unittest.mock import MagicMock
from mlflow.entities.model_registry import ModelVersion

from pynumaflow.function import Datum
from numalogic.models.autoencoder.variants import VanillaAE

from numaprom.constants import TESTS_DIR, ARGO_CD, ARGO_ROLLOUTS, METRIC_CONFIG
from numaprom.factory import HandlerFactory
from numaprom.tests import *

sys.modules["numaprom.mlflow"] = MagicMock()
MODEL_DIR = os.path.join(TESTS_DIR, "resources", "models")


def mockenv(**envvars):
    return mock.patch.dict(os.environ, envvars, clear=True)


def get_datum(data: str or bytes) -> Datum:
    if type(data) is not bytes:
        data = json.dumps(data).encode("utf-8")

    return Datum(value=data, event_time=datetime.datetime.now(), watermark=datetime.datetime.now())


def get_stream_data(data_path: str):
    with open(data_path) as fp:
        data = json.load(fp)
    return data


def get_prepoc_input(data_path: str) -> Datum:
    data = get_stream_data(data_path)
    output_ = None
    for obj in data:
        output_ = window("", get_datum(obj))
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


def return_mock_metric_config():
    return {
        "metric_1": {
            "keys": ["namespace", "name"],
            "model_config": {
                "name": ARGO_CD,
                "win_size": 12,
                "threshold_min": 0.1,
                "model_name": "ae_sparse",
                "retrain_freq_hr": 8,
                "resume_training": "True",
                "keys": ["namespace", "name"],
                "metrics": [
                    "metric_1",
                ]
            }

        },
        "metric_2": {
            "keys": ["namespace", "name", "hash_id"],
            "model_config": {
                "name": ARGO_ROLLOUTS,
                "win_size": 12,
                "threshold_min": 0.001,
                "model_name": "ae_sparse",
                "retrain_freq_hr": 8,
                "resume_training": "True",
                "keys": ["namespace", "name"],
                "metrics": [
                    "metric_2",
                ]
            }

        }
    }
