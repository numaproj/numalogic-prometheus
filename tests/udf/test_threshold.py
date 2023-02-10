import os
import unittest
from unittest.mock import patch, Mock

from freezegun import freeze_time
from numalogic.registry import MLflowRegistry
from orjson import orjson

from numaprom._constants import TESTS_DIR, METRIC_CONFIG
from numaprom.entities import Status, StreamPayload, TrainerPayload, Header
from tests.tools import (
    get_threshold_input,
    return_mock_metric_config,
    get_datum,
    return_threshold_clf,
)

# Make sure to import this in the end
from numaprom.udf import threshold

DATA_DIR = os.path.join(TESTS_DIR, "resources", "data")
STREAM_DATA_PATH = os.path.join(DATA_DIR, "stream.json")


@patch.dict(METRIC_CONFIG, return_mock_metric_config())
class TestThreshold(unittest.TestCase):
    @freeze_time("2022-02-20 12:00:00")
    @patch.object(MLflowRegistry, "load", Mock(return_value=return_threshold_clf()))
    def test_threshold(self):
        thresh_input = get_threshold_input(STREAM_DATA_PATH)
        assert thresh_input.items(), print("input items is empty", thresh_input)

        for msg in thresh_input.items():
            _in = get_datum(msg.value)
            _out = threshold("", _in)
            for _datum in _out.items():
                out_data = _datum.value.decode("utf-8")
                payload = StreamPayload(**orjson.loads(out_data))

                self.assertEqual(payload.status, Status.THRESHOLD)
                self.assertEqual(Header.MODEL_INFERENCE, payload.header)
                self.assertTrue(payload.win_arr)
                self.assertTrue(payload.win_ts_arr)

    @patch.object(MLflowRegistry, "load", Mock(return_value=None))
    def test_threshold_no_prev_clf(self):
        thresh_input = get_threshold_input(STREAM_DATA_PATH, prev_clf_exists=False)
        assert thresh_input.items(), print("input items is empty", thresh_input)

        for msg in thresh_input.items():
            _in = get_datum(msg.value)
            _out = threshold("", _in)
            out_data = _out.items()[0].value.decode("utf-8")
            train_payload = TrainerPayload(**orjson.loads(out_data))

            self.assertEqual(1, len(_out.items()))
            self.assertTrue(train_payload)
            self.assertEqual(Header.TRAIN_REQUEST, train_payload.header)
            self.assertIsInstance(train_payload, TrainerPayload)

    @freeze_time("2022-02-20 12:00:00")
    @patch.object(MLflowRegistry, "load", Mock(return_value=None))
    def test_threshold_no_clf(self):
        thresh_input = get_threshold_input(STREAM_DATA_PATH)
        assert thresh_input.items(), print("input items is empty", thresh_input)

        for msg in thresh_input.items():
            _in = get_datum(msg.value)
            _out = threshold("", _in)

            out_data = _out.items()[0].value.decode("utf-8")
            train_payload = TrainerPayload(**orjson.loads(out_data))
            self.assertTrue(train_payload)
            self.assertIsInstance(train_payload, TrainerPayload)
            self.assertEqual(Header.TRAIN_REQUEST, train_payload.header)

            out_data = _out.items()[1].value.decode("utf-8")
            stream_payload = StreamPayload(**orjson.loads(out_data))
            self.assertEqual(Header.STATIC_INFERENCE, stream_payload.header)
            self.assertIsInstance(stream_payload, StreamPayload)


if __name__ == "__main__":
    unittest.main()
