import os
import unittest
from unittest.mock import patch, Mock

from freezegun import freeze_time
from numalogic.models.autoencoder import AutoencoderTrainer
from numalogic.registry import RedisRegistry
from orjson import orjson
from pynumaflow.mapper import Messages

from numaprom._constants import TESTS_DIR
from numaprom.entities import Status, StreamPayload, Header
from tests import redis_client, inference
from tests.tools import (
    get_inference_input,
    return_mock_lstmae,
    get_datum,
    return_stale_model_redis,
)

DATA_DIR = os.path.join(TESTS_DIR, "resources", "data")
MODEL_DIR = os.path.join(TESTS_DIR, "resources", "models")
STREAM_DATA_PATH = os.path.join(DATA_DIR, "stream.json")


class TestInference(unittest.TestCase):
    inference_input: Messages = None

    @classmethod
    def setUpClass(cls) -> None:
        redis_client.flushall()
        cls.inference_input = get_inference_input(STREAM_DATA_PATH)
        assert len(cls.inference_input), print("input items is empty", cls.inference_input)

    def setUp(self) -> None:
        redis_client.flushall()

    @freeze_time("2022-02-20 12:00:00")
    @patch.object(RedisRegistry, "load", Mock(return_value=return_mock_lstmae()))
    def test_inference(self):
        for msg in self.inference_input:
            _in = get_datum(msg.value)
            _out = inference([""], _in)
            for _m in _out:
                out_data = _m.value.decode("utf-8")
                payload = StreamPayload(**orjson.loads(out_data))

                self.assertEqual(payload.status, Status.INFERRED)
                self.assertEqual(payload.header, Header.MODEL_INFERENCE)
                self.assertTrue(payload.win_arr)
                self.assertTrue(payload.win_ts_arr)

    @freeze_time("2022-02-20 12:00:00")
    @patch.object(RedisRegistry, "load", Mock(return_value=return_mock_lstmae()))
    @patch.object(AutoencoderTrainer, "predict", Mock(side_effect=RuntimeError))
    def test_inference_err(self):
        for msg in self.inference_input:
            _out = inference([""], get_datum(msg.value))
            for _m in _out:
                payload = StreamPayload(**orjson.loads(_m.value))
                self.assertEqual(payload.status, Status.RUNTIME_ERROR)
                self.assertEqual(payload.header, Header.STATIC_INFERENCE)
                self.assertTrue(payload.win_arr)
                self.assertTrue(payload.win_ts_arr)

    @patch.object(RedisRegistry, "load", Mock(return_value=None))
    def test_no_model(self):
        for msg in self.inference_input:
            _out = inference([""], get_datum(msg.value))
            payload = StreamPayload(**orjson.loads(_out[0].value))
            self.assertEqual(payload.status, Status.ARTIFACT_NOT_FOUND)
            self.assertEqual(payload.header, Header.STATIC_INFERENCE)
            self.assertIsInstance(payload, StreamPayload)

    @freeze_time("2022-02-20 12:00:00")
    @patch.object(RedisRegistry, "load", Mock(return_value=return_mock_lstmae()))
    def test_no_prev_model(self):
        inference_input = get_inference_input(STREAM_DATA_PATH, prev_clf_exists=False)
        assert len(inference_input), print("input items is empty", inference_input)
        for msg in inference_input:
            _in = get_datum(msg.value)
            _out = inference([""], _in)
            out_data = _out[0].value.decode("utf-8")
            payload = StreamPayload(**orjson.loads(out_data))
            self.assertEqual(payload.status, Status.ARTIFACT_NOT_FOUND)
            self.assertEqual(payload.header, Header.STATIC_INFERENCE)
            self.assertIsInstance(payload, StreamPayload)

    @patch.object(RedisRegistry, "load", Mock(return_value=return_stale_model_redis()))
    def test_stale_model(self):
        for msg in self.inference_input:
            _in = get_datum(msg.value)
            _out = inference([""], _in)
            for _datum in _out:
                payload = StreamPayload(**orjson.loads(_out[0].value))
                self.assertTrue(payload)
                self.assertEqual(payload.status, Status.INFERRED)
                self.assertEqual(payload.header, Header.MODEL_STALE)


if __name__ == "__main__":
    unittest.main()
