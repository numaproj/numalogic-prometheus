import json
import os
import socket
import unittest
from pprint import pprint
from unittest.mock import patch

from nlogicprom.constants import TESTS_DIR, DEFAULT_WIN_SIZE
from nlogicprom.entities import MetricType
from nlogicprom.tools import extract, get_data, is_host_reachable

DATA_DIR = os.path.join(TESTS_DIR, "resources", "data")
REQ_2xx = os.path.join(DATA_DIR, "2xx.txt")
STREAM_DATA_PATH = os.path.join(DATA_DIR, "stream.json")
ROLLOUTS_STREAM_PATH = os.path.join(DATA_DIR, "rollouts_stream.json")


def mock_resolver(*_, **__):
    raise socket.gaierror


class TestTools(unittest.TestCase):
    INFER_OUT = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.win_size = DEFAULT_WIN_SIZE
        with open(STREAM_DATA_PATH) as fp:
            cls.data = json.load(fp)
        with open(ROLLOUTS_STREAM_PATH) as fp:
            cls.rollouts_data = json.load(fp)

    def setUp(self) -> None:
        window = [
            {"timestamp": "1654121163989", "value": "2"},
            {"timestamp": "1654121168989", "value": "3"},
            {"timestamp": "1654121173989", "value": "4"},
        ]
        self.rollout_metric = self.rollouts_data[0]
        self.metric = self.data[-1]
        self.rollout_metric["window"] = window
        self.metric["window"] = window
        self.data_2xx = get_data(REQ_2xx)
        self.data_2xx["window"] = window

    def test_extract(self):
        out = extract(self.metric)
        pprint(out, indent=2)
        self.assertEqual(out.metric, MetricType.CPU.value)

    def test_extract_rollout_metric(self):
        out = extract(self.rollout_metric)
        pprint(out, indent=2)
        self.assertEqual(out.metric, MetricType.HASH_ERROR_RATE.value)
        self.assertEqual(out.hash_id, "64f9bb588")

    def test_extract_err(self):
        with self.assertRaises(NotImplementedError):
            extract(self.data_2xx)

    def test_is_host_reachable(self):
        self.assertTrue(is_host_reachable("google.com"))

    @patch("nlogicprom.tools.get_ipv4_by_hostname", mock_resolver)
    def test_is_host_reachable_err(self):
        self.assertFalse(is_host_reachable("google.com", max_retries=2, sleep_sec=1))
