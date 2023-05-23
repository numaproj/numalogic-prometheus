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
        assert cls.preproc_input.items(), print("input items is empty", cls.preproc_input)

    def setUp(self) -> None:
        redis_client.flushall()

    @patch.object(RedisRegistry, "load", Mock(return_value=return_preproc_clf()))
    def test_preprocess(self):
        for msg in self.preproc_input.items():
            _in = get_datum(msg.value)
            _out = preprocess("", _in)
            for _datum in _out.items():
                out_data = _datum.value.decode("utf-8")
                payload = StreamPayload(**orjson.loads(out_data))

                self.assertEqual(payload.status, Status.PRE_PROCESSED)
                self.assertEqual(payload.header, Header.MODEL_INFERENCE)
                self.assertTrue(payload.win_arr)
                self.assertTrue(payload.win_ts_arr)
                self.assertIsInstance(payload, StreamPayload)

    @patch.object(RedisRegistry, "load", Mock(return_value=None))
    def test_preprocess_no_clf(self):
        for msg in self.preproc_input.items():
            _in = get_datum(msg.value)
            _out = preprocess("", _in)
            out_data = _out.items()[0].value.decode("utf-8")
            payload = StreamPayload(**orjson.loads(out_data))
            self.assertEqual(payload.status, Status.ARTIFACT_NOT_FOUND)
            self.assertEqual(payload.header, Header.STATIC_INFERENCE)
            self.assertIsInstance(payload, StreamPayload)

    @patch.object(RedisRegistry, "load", Mock(return_value=return_preproc_clf()))
    def test_preprocess_with_nan(self):
        preproc_input = get_prepoc_input(STREAM_NAN_DATA_PATH)
        assert preproc_input.items(), print("input items is empty", preproc_input)

        for msg in preproc_input.items():
            _in = get_datum(msg.value)
            _out = preprocess("", _in)
            out_data = _out.items()[0].value.decode("utf-8")
            payload = StreamPayload(**orjson.loads(out_data))
            stream_arr = payload.get_stream_array()

            self.assertEqual(payload.status, Status.PRE_PROCESSED)
            self.assertEqual(payload.header, Header.MODEL_INFERENCE)
            self.assertTrue(np.isfinite(stream_arr).all())
            self.assertTrue(payload.win_ts_arr)
            self.assertIsInstance(payload, StreamPayload)


if __name__ == "__main__":
    unittest.main()
