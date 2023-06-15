import os
import unittest
from unittest.mock import patch, Mock

import numpy as np
from numalogic.registry import RedisRegistry
from orjson import orjson
from pynumaflow.function import Messages

from numaprom._constants import TESTS_DIR
from numaprom.entities import Status, StreamPayload, Header

# Make sure to import this in the end
from tests import redis_client, preprocess
from tests.tools import get_prepoc_input, get_datum, return_preproc_clf

DATA_DIR = os.path.join(TESTS_DIR, "resources", "data")
STREAM_DATA_PATH = os.path.join(DATA_DIR, "stream.json")
STREAM_NAN_DATA_PATH = os.path.join(DATA_DIR, "stream_nan.json")


class TestPreprocess(unittest.TestCase):
    preproc_input: Messages = None

    @classmethod
    def setUpClass(cls) -> None:
        redis_client.flushall()
        cls.preproc_input = get_prepoc_input(STREAM_DATA_PATH)
        assert len(cls.preproc_input), print("input items is empty", cls.preproc_input)

    def setUp(self) -> None:
        redis_client.flushall()

    @patch.object(RedisRegistry, "load", Mock(return_value=return_preproc_clf()))
    def test_preprocess(self):
        for msg in self.preproc_input:
            _out = preprocess([""], get_datum(msg.value))
            self.assertGreater(len(_out), 0)
            for _m in _out:
                payload = StreamPayload(**orjson.loads(_m.value))

                self.assertEqual(payload.status, Status.PRE_PROCESSED)
                self.assertEqual(payload.header, Header.MODEL_INFERENCE)
                self.assertTrue(payload.win_arr)
                self.assertTrue(payload.win_ts_arr)
                self.assertIsInstance(payload, StreamPayload)

    @patch.object(RedisRegistry, "load", Mock(return_value=None))
    def test_preprocess_no_clf(self):
        for msg in self.preproc_input:
            _in = get_datum(msg.value)
            _out = preprocess([""], _in)
            payload = StreamPayload(**orjson.loads(_out[0].value))
            self.assertEqual(payload.status, Status.ARTIFACT_NOT_FOUND)
            self.assertEqual(payload.header, Header.STATIC_INFERENCE)
            self.assertIsInstance(payload, StreamPayload)

    @patch.object(RedisRegistry, "load", Mock(return_value=return_preproc_clf()))
    def test_preprocess_with_nan(self):
        preproc_input = get_prepoc_input(STREAM_NAN_DATA_PATH)
        assert len(preproc_input), print("input items is empty", preproc_input)

        for msg in preproc_input:
            _in = get_datum(msg.value)
            _out = preprocess([""], _in)
            payload = StreamPayload(**orjson.loads(_out[0].value))
            stream_arr = payload.get_stream_array()

            self.assertNotEqual(id(stream_arr), payload.win_arr)
            self.assertEqual(payload.status, Status.PRE_PROCESSED)
            self.assertEqual(payload.header, Header.MODEL_INFERENCE)
            self.assertTrue(np.isfinite(stream_arr).all())
            self.assertTrue(payload.win_ts_arr)
            self.assertIsInstance(payload, StreamPayload)

    @patch.object(RedisRegistry, "load", Mock(side_effect=ModuleNotFoundError))
    def test_unhandled_exception(self):
        with self.assertRaises(Exception):
            for msg in self.preproc_input:
                _in = get_datum(msg.value)
                _out = preprocess([""], _in)
                payload = StreamPayload(**orjson.loads(_out[0].value))
                self.assertEqual(payload.status, Status.ARTIFACT_NOT_FOUND)
                self.assertEqual(payload.header, Header.STATIC_INFERENCE)
                self.assertIsInstance(payload, StreamPayload)


if __name__ == "__main__":
    unittest.main()
