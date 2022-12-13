import os
import json
import unittest
from unittest.mock import patch, Mock

import torch
from mlflow.entities.model_registry import ModelVersion
from numalogic.models.autoencoder.variants import VanillaAE
from numalogic.registry import MLflowRegistrar

from numaprom.tests import *
from numaprom.constants import TESTS_DIR
from numaprom.entities import Payload, Status
from numaprom.tests.tools import get_inference_input, return_mock_cpu_load, mockenv
from numaprom.udf.inference import inference

DATA_DIR = os.path.join(TESTS_DIR, "resources", "data")
MODEL_DIR = os.path.join(TESTS_DIR, "resources", "models")
STREAM_DATA_PATH = os.path.join(DATA_DIR, "stream.json")



def return_stale_model(*_, **__):
    return {
        "primary_artifact": VanillaAE(12),
        "metadata": torch.load(os.path.join(MODEL_DIR, "model_cpu.pth")),
        "model_properties": ModelVersion(
            creation_timestamp=1656615600000,
            current_stage="Production",
            description="",
            last_updated_timestamp=1656615600000,
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


class TestInference(unittest.TestCase):

    @classmethod
    @mockenv(WIN_SIZE="12")
    def setUpClass(cls) -> None:
        redis_client.flushall()
        cls.inference_input = get_inference_input(STREAM_DATA_PATH)

    @patch.object(MLflowRegistrar, "load", Mock(return_value=return_mock_cpu_load()))
    def test_inference(self):
        _out = inference("", self.inference_input)
        data = _out.items()[0]._value.decode("utf-8")
        payload = Payload.from_json(data)
        self.assertEqual(payload.status, Status.INFERRED)

    @patch.object(MLflowRegistrar, "load", Mock(return_value=None))
    def test_no_model(self):
        _out = inference("", self.inference_input)
        data = _out.items()[0]._value.decode("utf-8")
        data = json.loads(data)

    @patch.object(MLflowRegistrar, "load", Mock(return_value=return_stale_model()))
    def test_stale_model(self):
        _out = inference("", self.inference_input)
        train_payload = json.loads(_out.items()[0]._value.decode("utf-8"))
        postprocess_payload = Payload.from_json(_out.items()[1]._value.decode("utf-8"))
        self.assertEqual(postprocess_payload.status, Status.INFERRED)
