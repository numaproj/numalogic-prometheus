import json
import os
import unittest
from unittest.mock import patch

from numaprom.constants import TESTS_DIR, METRIC_CONFIG
from numaprom.tests.tools import get_stream_data, mockenv, get_datum
from numaprom.udf import metric_filter

DATA_DIR = os.path.join(TESTS_DIR, "resources", "data")
STREAM_DATA_PATH = os.path.join(DATA_DIR, "stream.json")


class TestFilter(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.input_stream = get_stream_data(STREAM_DATA_PATH)

    @patch.dict(METRIC_CONFIG, {"metric_1": []})
    @mockenv(LABEL="numalogic", LABEL_VALUES='["true"]')
    def test_filter(self):
        for data in self.input_stream:
            _out = metric_filter("", get_datum(data))
            _out = _out.items()[0].value.decode("utf-8")
            if _out:
                data = json.loads(_out)
                self.assertEqual(data["labels"]["numalogic"], "true")


if __name__ == "__main__":
    unittest.main()
