import json
import os
import unittest
from unittest.mock import patch
from pynumaflow.function._dtypes import DROP

from nlogicprom.constants import TESTS_DIR
from nlogicprom.tests import *
from nlogicprom.tests.tools import get_datum
from nlogicprom.tools import decode_msg

DATA_DIR = os.path.join(TESTS_DIR, "resources", "data")
STREAM_DATA_PATH = os.path.join(DATA_DIR, "stream.json")
ROLLOUTS_STREAM_PATH = os.path.join(DATA_DIR, "rollouts_stream.json")


@patch.dict("os.environ", {"WIN_SIZE": "3"}, clear=True)
class TestWindow(unittest.TestCase):
    data = None

    @classmethod
    def setUpClass(cls) -> None:
        with open(STREAM_DATA_PATH) as fp:
            cls.data = json.load(fp)
        with open(ROLLOUTS_STREAM_PATH) as fp:
            cls.rollouts_data = json.load(fp)

    def tearDown(self) -> None:
        redis_client.flushall()

    def test_window(self):
        outs = []
        for idx, data in enumerate(self.data):
            out = window(None, get_datum(data))
            outs.append(out)
        data = decode_msg(outs[-1].items()[0].value)
        self.assertListEqual(
            [10, 11, 12], list(map(lambda x: int(x["value"]), data["processedMetrics"]))
        )

    def test_window_rollouts(self):
        outs = []
        for idx, data in enumerate(self.rollouts_data):
            out = window(None, get_datum(data))
            outs.append(out)
        data = decode_msg(outs[-1].items()[0].value)
        self.assertEqual(data["hash_id"], "64f9bb588")

    @patch.dict("os.environ", {"BUFF_SIZE": "2"}, clear=True)
    def test_window_err(self):
        for data in self.data:
            out = window(None, get_datum(data))
            self.assertIsNone(out)

    def test_window_drop(self):
        for _d in self.data:
            out = window(None, get_datum(_d))
            self.assertEqual(DROP, out.items()[0].key)
            break


if __name__ == "__main__":
    unittest.main()
