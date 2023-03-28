import os
import unittest
from orjson import orjson
from freezegun import freeze_time
from unittest.mock import patch, Mock

from numalogic.registry import MLflowRegistry

from numaprom import tools
from numaprom._constants import TESTS_DIR
from numaprom.entities import Status, StreamPayload, TrainerPayload, Header
from tests import redis_client
from tests.tools import (
    get_threshold_input,
    get_datum,
    return_threshold_clf,
    mock_configs,
)

# Make sure to import this in the end
from numaprom.udf import threshold

DATA_DIR = os.path.join(TESTS_DIR, "resources", "data")
STREAM_DATA_PATH = os.path.join(DATA_DIR, "stream.json")


@patch("numaprom.tools.set_aws_session", Mock(return_value=None))
@patch.object(tools, "get_all_configs", Mock(return_value=mock_configs()))
class TestThreshold(unittest.TestCase):
    def setUp(self) -> None:
        redis_client.flushall()

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
                self.assertEqual(payload.header, Header.MODEL_INFERENCE)
                self.assertTrue(payload.win_arr)
                self.assertTrue(payload.win_ts_arr)

    @patch.object(MLflowRegistry, "load", Mock(return_value=return_threshold_clf()))
    def test_threshold_prev_stale_model(self):
        thresh_input = get_threshold_input(STREAM_DATA_PATH, prev_model_stale=True)
        assert thresh_input.items(), print("input items is empty", thresh_input)

        for msg in thresh_input.items():
            _in = get_datum(msg.value)
            _out = threshold("", _in)
            train_payload = TrainerPayload(**orjson.loads(_out.items()[0].value.decode("utf-8")))
            payload = StreamPayload(**orjson.loads(_out.items()[1].value.decode("utf-8")))
            self.assertEqual(payload.header, Header.MODEL_STALE)
            self.assertEqual(payload.status, Status.THRESHOLD)
            self.assertIsInstance(train_payload, TrainerPayload)

    @patch.object(MLflowRegistry, "load", Mock(return_value=None))
    def test_threshold_no_prev_clf(self):
        thresh_input = get_threshold_input(STREAM_DATA_PATH, prev_clf_exists=False)
        assert thresh_input.items(), print("input items is empty", thresh_input)

        for msg in thresh_input.items():
            _in = get_datum(msg.value)
            _out = threshold("", _in)
            train_payload = TrainerPayload(**orjson.loads(_out.items()[0].value.decode("utf-8")))
            payload = StreamPayload(**orjson.loads(_out.items()[1].value.decode("utf-8")))
            self.assertEqual(payload.header, Header.STATIC_INFERENCE)
            self.assertEqual(payload.status, Status.ARTIFACT_NOT_FOUND)
            self.assertIsInstance(train_payload, TrainerPayload)

    @freeze_time("2022-02-20 12:00:00")
    @patch.object(MLflowRegistry, "load", Mock(return_value=None))
    def test_threshold_no_clf(self):
        thresh_input = get_threshold_input(STREAM_DATA_PATH)
        print(thresh_input)
        assert thresh_input.items(), print("input items is empty", thresh_input)

        for msg in thresh_input.items():
            _in = get_datum(msg.value)
            _out = threshold("", _in)

            train_payload = TrainerPayload(**orjson.loads(_out.items()[0].value.decode("utf-8")))
            payload = StreamPayload(**orjson.loads(_out.items()[1].value.decode("utf-8")))
            self.assertEqual(payload.header, Header.STATIC_INFERENCE)
            self.assertEqual(payload.status, Status.ARTIFACT_NOT_FOUND)
            self.assertIsInstance(train_payload, TrainerPayload)


if __name__ == "__main__":
    unittest.main()
