import os
import unittest
from unittest.mock import patch

from nlogicprom.constants import TESTS_DIR
from nlogicprom.entities import Payload, Status
from nlogicprom.tests.tools import get_prepoc_input
from nlogicprom.udf.preprocess import preprocess

DATA_DIR = os.path.join(TESTS_DIR, "resources", "data")
STREAM_DATA_PATH = os.path.join(DATA_DIR, "stream.json")
ROLLOUTS_STREAM_PATH = os.path.join(DATA_DIR, "rollouts_stream.json")


class TestPreprocess(unittest.TestCase):
    preproc_input = None
    rollouts_preproc_input = None

    @classmethod
    @patch.dict("os.environ", {"WIN_SIZE": "3"})
    def setUpClass(cls) -> None:
        cls.preproc_input = get_prepoc_input(STREAM_DATA_PATH)
        cls.rollouts_preproc_input = get_prepoc_input(ROLLOUTS_STREAM_PATH)

    def test_preprocess1(self):
        _out = preprocess(None, self.preproc_input)
        data = _out.items()[0]._value.decode("utf-8")
        payload = Payload.from_json(data)
        self.assertEqual(payload.status, Status.PRE_PROCESSED)

    def test_preprocess_rollouts(self):
        _out = preprocess(None, self.rollouts_preproc_input)
        data = _out.items()[0]._value.decode("utf-8")
        payload = Payload.from_json(data)
        self.assertEqual(payload.status, Status.PRE_PROCESSED)
        self.assertEqual(payload.hash_id, "64f9bb588")


if __name__ == "__main__":
    unittest.main()
