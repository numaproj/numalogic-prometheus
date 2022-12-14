import os
import json
import unittest
from unittest.mock import patch, Mock

from numalogic.registry import MLflowRegistrar

from numaprom.tests import *
from numaprom.constants import TESTS_DIR, METRIC_CONFIG
from numaprom.entities import Payload, Status
from numaprom.tests.tools import (
    get_inference_input,
    return_mock_metric_config,
    return_stale_model,
    return_mock_lstmae
)
from numaprom.udf.inference import inference

DATA_DIR = os.path.join(TESTS_DIR, "resources", "data")
MODEL_DIR = os.path.join(TESTS_DIR, "resources", "models")
STREAM_DATA_PATH = os.path.join(DATA_DIR, "stream.json")


@patch.dict(METRIC_CONFIG, return_mock_metric_config())
class TestInference(unittest.TestCase):

    @classmethod
    @patch.dict(METRIC_CONFIG, return_mock_metric_config())
    def setUpClass(cls) -> None:
        redis_client.flushall()
        cls.inference_input = get_inference_input(STREAM_DATA_PATH)

    @patch.object(MLflowRegistrar, "load", Mock(return_value=return_mock_lstmae()))
    def test_inference(self):
        _out = inference("", self.inference_input)
        data = _out.items()[0]._value.decode("utf-8")
        payload = Payload.from_json(data)
        self.assertEqual(payload.status, Status.INFERRED)

    @patch.object(MLflowRegistrar, "load", Mock(return_value=None))
    def test_no_model(self):
        _out = inference("", self.inference_input)
        train_payload = json.loads(_out.items()[0]._value.decode("utf-8"))
        self.assertFalse(train_payload["resume_training"])

    @patch.object(MLflowRegistrar, "load", Mock(return_value=return_stale_model()))
    def test_stale_model(self):
        _out = inference("", self.inference_input)
        train_payload = json.loads(_out.items()[0]._value.decode("utf-8"))
        postprocess_payload = Payload.from_json(_out.items()[1]._value.decode("utf-8"))
        self.assertEqual(postprocess_payload.status, Status.INFERRED)
        self.assertTrue(train_payload["resume_training"])
