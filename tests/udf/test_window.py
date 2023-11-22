import os
import orjson
import unittest

from pynumaflow.mapper._dtypes import DROP

from numaprom._constants import TESTS_DIR
from numaprom.entities import StreamPayload
from tests.tools import get_datum, get_stream_data, mockenv
from tests import redis_client, window

DATA_DIR = os.path.join(TESTS_DIR, "resources", "data")
STREAM_DATA_PATH = os.path.join(DATA_DIR, "stream.json")


class TestWindow(unittest.TestCase):
    input_stream: dict = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.input_stream = get_stream_data(STREAM_DATA_PATH)
        assert len(cls.input_stream), print("input items is empty", cls.input_stream)

    def tearDown(self) -> None:
        redis_client.flushall()

    def test_window(self):
        for idx, data in enumerate(self.input_stream):
            _out = window([""], get_datum(data))
            if len(_out[0].tags) > 0:
                if not _out[0].tags[0] == DROP:
                    _out = _out[0].value.decode("utf-8")
                    payload = StreamPayload(**orjson.loads(_out))
                    self.assertTrue(payload)

    def test_window_duplicate_element(self):
        uuids = set()
        for idx, data in enumerate(self.input_stream[-3:]):
            _out = window([""], get_datum(data))
            if len(_out[0].tags) > 0 and _out[0].tags[0] == DROP:
                continue
            else:
                _out = _out[0].value.decode("utf-8")
                payload = StreamPayload(**orjson.loads(_out))
                uuids.add(payload.uuid)
                self.assertTrue(payload)
        self.assertEqual(1, len(uuids))

    @mockenv(BUFF_SIZE="1")
    def test_window_err(self):
        with self.assertRaises(ValueError):
            for data in self.input_stream:
                window([""], get_datum(data))

    def test_window_drop(self):
        for _d in self.input_stream:
            out = window([""], get_datum(_d))
            self.assertEqual(DROP, out[0].tags[0])
            break


if __name__ == "__main__":
    unittest.main()
