import os
import unittest
from unittest.mock import patch, Mock

from freezegun import freeze_time
from numalogic.registry import MLflowRegistry
from orjson import orjson

from numaprom._constants import TESTS_DIR
from numaprom.entities import Status, StreamPayload, Header
from numaprom.udf.inference import inference
from numaprom.watcher import ConfigManager
from tests import redis_client
from tests.tools import (
    get_inference_input,
    return_stale_model,
    return_mock_lstmae,
    get_datum,
    mock_configs,
)

DATA_DIR = os.path.join(TESTS_DIR, "resources", "data")
MODEL_DIR = os.path.join(TESTS_DIR, "resources", "models")
STREAM_DATA_PATH = os.path.join(DATA_DIR, "stream.json")


@patch("numaprom.tools.set_aws_session", Mock(return_value=None))
@patch.object(ConfigManager, "load_configs", Mock(return_value=mock_configs()))
class TestInference(unittest.TestCase):
    @classmethod
    @patch("numaprom.tools.set_aws_session", Mock(return_value=None))
    @patch.object(ConfigManager, "load_configs", Mock(return_value=mock_configs()))
    def setUpClass(cls) -> None:
        redis_client.flushall()
        cls.inference_input = get_inference_input(STREAM_DATA_PATH)
        assert cls.inference_input.items(), print("input items is empty", cls.inference_input)

    @freeze_time("2022-02-20 12:00:00")
    @patch.object(MLflowRegistry, "load", Mock(return_value=return_mock_lstmae()))
    def test_inference(self):
        for msg in self.inference_input.items():
            _in = get_datum(msg.value)
            _out = inference("", _in)
            for _datum in _out.items():
                out_data = _datum.value.decode("utf-8")
                payload = StreamPayload(**orjson.loads(out_data))

                self.assertEqual(payload.status, Status.INFERRED)
                self.assertEqual(payload.header, Header.MODEL_INFERENCE)
                self.assertTrue(payload.win_arr)
                self.assertTrue(payload.win_ts_arr)

    @patch.object(MLflowRegistry, "load", Mock(return_value=None))
    def test_no_model(self):
        for msg in self.inference_input.items():
            _in = get_datum(msg.value)
            _out = inference("", _in)
            out_data = _out.items()[0].value.decode("utf-8")
            payload = StreamPayload(**orjson.loads(out_data))
            self.assertEqual(payload.status, Status.ARTIFACT_NOT_FOUND)
            self.assertEqual(payload.header, Header.STATIC_INFERENCE)
            self.assertIsInstance(payload, StreamPayload)

    @freeze_time("2022-02-20 12:00:00")
    @patch.object(MLflowRegistry, "load", Mock(return_value=return_mock_lstmae()))
    def test_no_prev_model(self):
        inference_input = get_inference_input(STREAM_DATA_PATH, prev_clf_exists=False)
        assert self.inference_input.items(), print("input items is empty", self.inference_input)
        for msg in inference_input.items():
            _in = get_datum(msg.value)
            _out = inference("", _in)
            out_data = _out.items()[0].value.decode("utf-8")
            payload = StreamPayload(**orjson.loads(out_data))
            self.assertEqual(payload.status, Status.ARTIFACT_NOT_FOUND)
            self.assertEqual(payload.header, Header.STATIC_INFERENCE)
            self.assertIsInstance(payload, StreamPayload)

    @patch.object(MLflowRegistry, "load", Mock(return_value=return_stale_model()))
    def test_stale_model(self):
        for msg in self.inference_input.items():
            _in = get_datum(msg.value)
            _out = inference("", _in)
            for _datum in _out.items():
                payload = StreamPayload(**orjson.loads(_out.items()[0].value.decode("utf-8")))
                self.assertTrue(payload)
                self.assertEqual(payload.status, Status.INFERRED)
                self.assertEqual(payload.header, Header.MODEL_STALE)
