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
    def setUpClass(cls) -> None:
        cls.input_stream = get_stream_data(STREAM_DATA_PATH)

    def tearDown(self) -> None:
        redis_client.flushall()

    def test_window(self):
        for idx, data in enumerate(self.input_stream):
            _out = window("", get_datum(data))
            if not _out.items()[0].key == DROP:
                _out = _out.items()[0].value.decode("utf-8")
                payload = Payload.from_json(_out)
                keys = list(payload.key_map.values())
                if "metric_2" in keys:
                    self.assertEqual(keys, ["sandbox_numalogic_demo", "metric_2", "123456789"])
                if "metric_1" in keys:
                    self.assertEqual(keys, ["sandbox_numalogic_demo", "metric_1"])

    @mockenv(BUFF_SIZE="1")
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
