import os
import json
import torch
import unittest
from unittest.mock import patch, Mock

from mlflow.entities.model_registry import ModelVersion
from numalogic.models.autoencoder.variants import VanillaAE
from numalogic.registry import MLflowRegistrar

from nlogicprom.constants import TESTS_DIR
from nlogicprom.entities import Payload, Status, MetricType
from nlogicprom.tests.tools import get_inference_input, return_mock_cpu_load
from nlogicprom.udf.inference import inference
from nlogicprom.tests import *

DATA_DIR = os.path.join(TESTS_DIR, "resources", "data")
MODEL_DIR = os.path.join(TESTS_DIR, "resources", "models")
STREAM_DATA_PATH = os.path.join(DATA_DIR, "stream.json")
ROLLOUTS_STREAM_PATH = os.path.join(DATA_DIR, "rollouts_stream.json")


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


@patch.dict("os.environ", {"WIN_SIZE": "12"})
class TestInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        redis_client.flushall()
        cls.inference_input = get_inference_input(STREAM_DATA_PATH)
        cls.rollouts_inference_input = get_inference_input(ROLLOUTS_STREAM_PATH)

    @patch.object(MLflowRegistrar, "load", Mock(return_value=return_mock_cpu_load()))
    def test_inference(self):
        _out = inference(None, self.inference_input)
        data = _out.items()[0]._value.decode("utf-8")
        payload = Payload.from_json(data)
        self.assertEqual(payload.status, Status.INFERRED)

    @patch.object(MLflowRegistrar, "load", Mock(return_value=return_mock_cpu_load()))
    def test_inference_rollouts(self):
        _out = inference(None, self.rollouts_inference_input)
        data = _out.items()[0]._value.decode("utf-8")
        payload = Payload.from_json(data)
        self.assertEqual(payload.status, Status.INFERRED)
        self.assertEqual(payload.hash_id, "64f9bb588")

    @patch.object(MLflowRegistrar, "load", Mock(return_value=None))
    def test_no_model(self):
        _out = inference(None, self.inference_input)
        data = _out.items()[0]._value.decode("utf-8")
        data = json.loads(data)
        self.assertEqual(data["metric"], MetricType.CPU.value)

    @patch.object(MLflowRegistrar, "load", Mock(return_value=return_stale_model()))
    def test_stale_model(self):
        _out = inference(None, self.inference_input)
        train_payload = json.loads(_out.items()[0]._value.decode("utf-8"))
        postprocess_payload = Payload.from_json(_out.items()[1]._value.decode("utf-8"))
        self.assertEqual(train_payload["metric"], MetricType.CPU.value)
        self.assertEqual(postprocess_payload.status, Status.INFERRED)
