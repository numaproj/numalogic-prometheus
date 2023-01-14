import json
import os
import unittest
from unittest.mock import patch, Mock
from freezegun import freeze_time

from numalogic.registry import MLflowRegistry
from orjson import orjson

from numaprom._constants import TESTS_DIR, METRIC_CONFIG
from numaprom.entities import Status, StreamPayload
from tests import redis_client
from tests.tools import (
    get_inference_input,
    return_mock_metric_config,
    return_stale_model,
    return_mock_lstmae,
    get_datum,
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

    @freeze_time("2022-02-20 12:00:00")
    @patch.object(MLflowRegistry, "load", Mock(return_value=return_mock_lstmae()))
    def test_inference(self):
        for msg in self.inference_input.items():
            _in = get_datum(msg.value)
            _out = inference("", _in)
            out_data = _out.items()[0].value.decode("utf-8")
            payload = StreamPayload(**orjson.loads(out_data))

            self.assertEqual(payload.status, Status.INFERRED)
            self.assertTrue(payload.win_arr)
            self.assertTrue(payload.win_ts_arr)

    @patch.object(MLflowRegistry, "load", Mock(return_value=None))
    def test_no_model(self):
        for msg in self.inference_input.items():
            _in = get_datum(msg.value)
            _out = inference("", _in)
            train_payload = json.loads(_out.items()[0].value.decode("utf-8"))
            self.assertFalse(train_payload["resume_training"])

    @patch.object(MLflowRegistry, "load", Mock(return_value=return_stale_model()))
    def test_stale_model(self):
        for msg in self.inference_input.items():
            _in = get_datum(msg.value)
            _out = inference("", _in)
            train_payload = json.loads(_out.items()[0].value.decode("utf-8"))
            postprocess_payload = StreamPayload(
                **orjson.loads(_out.items()[1].value.decode("utf-8"))
            )

            self.assertTrue(train_payload)
            self.assertTrue(postprocess_payload)
            self.assertEqual(postprocess_payload.status, Status.INFERRED)
