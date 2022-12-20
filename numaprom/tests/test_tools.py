import os
import socket
import unittest
from pprint import pprint
from unittest.mock import patch

from numaprom.constants import TESTS_DIR, METRIC_CONFIG
from numaprom.entities import Payload
from numaprom.tests.tools import get_stream_data, return_mock_metric_config
from numaprom.tools import extract, is_host_reachable

DATA_DIR = os.path.join(TESTS_DIR, "resources", "data")
REQ_2xx = os.path.join(DATA_DIR, "2xx.txt")
STREAM_DATA_PATH = os.path.join(DATA_DIR, "stream.json")


def mock_resolver(*_, **__):
    raise socket.gaierror


@patch.dict(METRIC_CONFIG, return_mock_metric_config())
class TestTools(unittest.TestCase):
    INFER_OUT = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.input_stream = get_stream_data(STREAM_DATA_PATH)
        cls.window = [
            {"timestamp": "1654121163989", "value": "2"},
            {"timestamp": "1654121168989", "value": "3"},
            {"timestamp": "1654121173989", "value": "4"},
        ]

    def test_extract(self):
        for idx, data in enumerate(self.input_stream):
            data["window"] = self.window
            out = extract(data)
            pprint(out, indent=2)
            self.assertTrue(type(out), Payload)

    def test_is_host_reachable(self):
        self.assertTrue(is_host_reachable("google.com"))

    @patch("numaprom.tools.get_ipv4_by_hostname", mock_resolver)
    def test_is_host_reachable_err(self):
        self.assertFalse(is_host_reachable("google.com", max_retries=2, sleep_sec=1))
