import os
import unittest
from unittest.mock import patch, Mock

from numalogic.registry import MLflowRegistry
from orjson import orjson

from numaprom._constants import TESTS_DIR, METRIC_CONFIG
from numaprom.entities import Status, StreamPayload, TrainerPayload
from tests.tools import get_prepoc_input, return_mock_metric_config, get_datum, return_preproc_clf

# Make sure to import this in the end
from numaprom.udf.preprocess import preprocess

DATA_DIR = os.path.join(TESTS_DIR, "resources", "data")
STREAM_DATA_PATH = os.path.join(DATA_DIR, "stream.json")


class TestPreprocess(unittest.TestCase):
    preproc_input = None

    @classmethod
    @patch.dict(METRIC_CONFIG, return_mock_metric_config())
    def setUpClass(cls) -> None:
        cls.preproc_input = get_prepoc_input(STREAM_DATA_PATH)
        assert cls.preproc_input.items(), print("input items is empty", cls.preproc_input)

    @patch.object(MLflowRegistry, "load", Mock(return_value=return_preproc_clf()))
    def test_preprocess(self):
        for msg in self.preproc_input.items():
            _in = get_datum(msg.value)
            _out = preprocess("", _in)
            out_data = _out.items()[0].value.decode("utf-8")
            payload = StreamPayload(**orjson.loads(out_data))

            self.assertEqual(payload.status, Status.PRE_PROCESSED)
            self.assertTrue(payload.win_arr)
            self.assertTrue(payload.win_ts_arr)
            self.assertIsInstance(payload, StreamPayload)

    @patch.object(MLflowRegistry, "load", Mock(return_value=None))
    def test_preprocess_no_clf(self):
        for msg in self.preproc_input.items():
            _in = get_datum(msg.value)
            _out = preprocess("", _in)
            out_data = _out.items()[0].value.decode("utf-8")
            train_payload = TrainerPayload(**orjson.loads(out_data))
            self.assertTrue(train_payload)
            self.assertIsInstance(train_payload, TrainerPayload)


if __name__ == "__main__":
    unittest.main()
