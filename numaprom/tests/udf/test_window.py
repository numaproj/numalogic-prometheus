import os
import unittest
from unittest.mock import patch

from pynumaflow.function._dtypes import DROP

from numaprom.tests import *
from numaprom.entities import Payload
from numaprom.constants import TESTS_DIR, METRIC_CONFIG
from numaprom.tests.tools import get_datum, get_stream_data, mockenv, return_mock_metric_config

DATA_DIR = os.path.join(TESTS_DIR, "resources", "data")
STREAM_DATA_PATH = os.path.join(DATA_DIR, "stream.json")


@patch.dict(METRIC_CONFIG, return_mock_metric_config())
class TestWindow(unittest.TestCase):

    @classmethod
    @mockenv(WIN_SIZE="3")
    def setUpClass(cls) -> None:
        cls.input_stream = get_stream_data(STREAM_DATA_PATH)

    def tearDown(self) -> None:
        redis_client.flushall()

    def test_window(self):
        for idx, data in enumerate(self.input_stream):
            _out = window("", get_datum(data))
            _out = _out._messages[0]._value.decode("utf-8")
            if _out:
                payload = Payload.from_json(_out)
                if "metric_2" in payload.keys:
                    self.assertEqual(payload.keys, ['namespace_1', 'metric_2', '64f9bb588'])
                if "metric_1" in payload.keys:
                    self.assertEqual(payload.keys, ['namespace_1', 'metric_1'])

    @mockenv(BUFF_SIZE="2")
    def test_window_err(self):
        for data in self.input_stream:
            out = window("", get_datum(data))
            self.assertIsNone(out)

    def test_window_drop(self):
        for _d in self.input_stream:
            out = window("", get_datum(_d))
            self.assertEqual(DROP, out.items()[0].key)
            break


if __name__ == "__main__":
    unittest.main()
