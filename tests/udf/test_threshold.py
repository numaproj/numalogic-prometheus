import os
import unittest
from unittest.mock import patch, Mock

from numalogic.registry import MLflowRegistry
from orjson import orjson

from numaprom._constants import TESTS_DIR, METRIC_CONFIG
from numaprom.entities import Status, StreamPayload
from tests.tools import get_prepoc_input, return_mock_metric_config, get_datum, return_threshold_clf

# Make sure to import this in the end
from numaprom.udf import threshold

DATA_DIR = os.path.join(TESTS_DIR, "resources", "data")
STREAM_DATA_PATH = os.path.join(DATA_DIR, "stream.json")


class TestThreshold(unittest.TestCase):
    thresh_input = None

    @classmethod
    @patch.dict(METRIC_CONFIG, return_mock_metric_config())
    def setUpClass(cls) -> None:
        cls.thresh_input = get_prepoc_input(STREAM_DATA_PATH)
        assert cls.thresh_input.items(), print("input items is empty", cls.thresh_input)

    @patch.object(MLflowRegistry, "load", Mock(return_value=return_threshold_clf()))
    def test_threshold(self):
        for msg in self.thresh_input.items():
            _in = get_datum(msg.value)
            _out = threshold("", _in)
            out_data = _out.items()[0].value.decode("utf-8")
            payload = StreamPayload(**orjson.loads(out_data))

            self.assertEqual(payload.status, Status.THRESHOLD)
            self.assertTrue(payload.win_arr)
            self.assertTrue(payload.win_ts_arr)

    @patch.object(MLflowRegistry, "load", Mock(return_value=None))
    def test_threshold_no_clf(self):
        for msg in self.thresh_input.items():
            _in = get_datum(msg.value)
            _out = threshold("", _in)
            out_data = _out.items()[0].value.decode("utf-8")
            train_payload = orjson.loads(out_data)
            self.assertTrue(train_payload)


if __name__ == "__main__":
    unittest.main()
