import os
import unittest
from unittest.mock import patch

from numaprom.constants import TESTS_DIR, METRIC_CONFIG
from numaprom.entities import Payload, Status
from numaprom.tests.tools import get_prepoc_input, return_mock_metric_config
from numaprom.udf.preprocess import preprocess

DATA_DIR = os.path.join(TESTS_DIR, "resources", "data")
STREAM_DATA_PATH = os.path.join(DATA_DIR, "stream.json")


class TestPreprocess(unittest.TestCase):
    preproc_input = None

    @classmethod
    @patch.dict(METRIC_CONFIG, return_mock_metric_config())
    def setUpClass(cls) -> None:
        cls.preproc_input = get_prepoc_input(STREAM_DATA_PATH)

    def test_preprocess1(self):
        _out = preprocess("", self.preproc_input)
        data = _out.items()[0].value.decode("utf-8")
        payload = Payload.from_json(data)
        self.assertEqual(payload.status, Status.PRE_PROCESSED)


if __name__ == "__main__":
    unittest.main()
