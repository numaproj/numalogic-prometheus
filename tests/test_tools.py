import os
import socket
import unittest
from collections import OrderedDict
from unittest.mock import patch

import numpy as np

from numaprom._constants import TESTS_DIR
from numaprom.entities import StreamPayload
from numaprom.tools import is_host_reachable, WindowScorer
from numaprom.watcher import ConfigManager

DATA_DIR = os.path.join(TESTS_DIR, "resources", "data")
REQ_2xx = os.path.join(DATA_DIR, "2xx.txt")
STREAM_DATA_PATH = os.path.join(DATA_DIR, "stream.json")


def mock_resolver(*_, **__):
    raise socket.gaierror


class TestTools(unittest.TestCase):
    INFER_OUT = None

    def test_is_host_reachable(self):
        self.assertTrue(is_host_reachable("google.com"))

    @patch("numaprom.tools.get_ipv4_by_hostname", mock_resolver)
    def test_is_host_reachable_err(self):
        self.assertFalse(is_host_reachable("google.com", max_retries=2, sleep_sec=1))


class TestWindowScorer(unittest.TestCase):
    def test_get_winscore(self):
        metric_conf = ConfigManager().get_metric_config(
            {
                "name": "namespace_app_rollouts_http_request_error_rate",
                "namespace": "sandbox_numalogic_demo2",
            }
        )
        stream = np.random.uniform(low=1, high=2, size=(10, 1))
        payload = StreamPayload(
            uuid="123",
            composite_keys=OrderedDict({"namespace": "sandbox_numalogic_demo2"}),
            win_raw_arr=stream,
            win_arr=stream.copy(),
            win_ts_arr=list(map(str, range(10))),
        )
        winscorer = WindowScorer(metric_conf)
        final_score = winscorer.get_final_winscore(payload)
        self.assertLess(final_score, 3.0)
