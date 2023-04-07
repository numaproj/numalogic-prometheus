import os
import socket
import time
import unittest
from collections import OrderedDict
from unittest.mock import patch, Mock

import numpy as np

from numaprom import tools, watcher
from numaprom._constants import TESTS_DIR
from numaprom.entities import StreamPayload
from numaprom.tools import (
    is_host_reachable,
    get_metric_config,
    get_app_config,
    get_unified_config,
    WindowScorer,
)
from tests.tools import mock_configs

DATA_DIR = os.path.join(TESTS_DIR, "resources", "data")
REQ_2xx = os.path.join(DATA_DIR, "2xx.txt")
STREAM_DATA_PATH = os.path.join(DATA_DIR, "stream.json")


def mock_resolver(*_, **__):
    raise socket.gaierror


@patch.object(watcher, "load_configs", Mock(return_value=mock_configs()))
class TestTools(unittest.TestCase):
    INFER_OUT = None

    def test_is_host_reachable(self):
        self.assertTrue(is_host_reachable("google.com"))

    @patch("numaprom.tools.get_ipv4_by_hostname", mock_resolver)
    def test_is_host_reachable_err(self):
        self.assertFalse(is_host_reachable("google.com", max_retries=2, sleep_sec=1))

    def test_get_app_config(self):
        # from given config
        app_config = get_app_config(
            metric="rollout_latency", namespace="sandbox_numalogic_demo1"
        )
        self.assertTrue(app_config)
        self.assertEqual(app_config.namespace, "sandbox_numalogic_demo1")

        # from given default config
        app_config = get_app_config(
            metric="namespace_app_rollouts_http_request_error_rate", namespace="abc"
        )
        self.assertTrue(app_config)
        self.assertEqual(app_config.namespace, "default-argorollouts")
        app_config = get_app_config(
            metric="namespace_app_http_server_requests_error_rate", namespace="abc"
        )
        self.assertTrue(app_config)
        self.assertEqual(app_config.namespace, "default-argocd")

        # default config
        service_config = get_app_config(metric="random", namespace="abc")
        self.assertTrue(service_config)
        self.assertEqual(service_config.namespace, "default")

    def test_get_metric_config(self):
        # from given config
        metric_config = get_metric_config(
            metric="rollout_latency", namespace="sandbox_numalogic_demo1"
        )
        self.assertTrue(metric_config)
        self.assertEqual(metric_config.metric, "rollout_latency")

        # from given default config
        metric_config = get_metric_config(
            metric="namespace_app_rollouts_http_request_error_rate", namespace="abc"
        )
        self.assertTrue(metric_config)
        self.assertEqual(metric_config.metric, "namespace_app_rollouts_http_request_error_rate")

        # default config
        metric_config = get_metric_config(metric="random", namespace="abc")
        self.assertTrue(metric_config)
        self.assertEqual(metric_config.metric, "default")

    def test_get_unified_config(self):
        # from given config
        unified_config = get_unified_config(
            metric="rollout_latency", namespace="sandbox_numalogic_demo1"
        )
        self.assertTrue(unified_config)
        self.assertTrue("rollout_latency" in unified_config.unified_metrics)

        # from given default config
        unified_config = get_unified_config(
            metric="namespace_app_rollouts_http_request_error_rate", namespace="abc"
        )
        self.assertTrue(unified_config)
        self.assertTrue(
            "namespace_app_rollouts_http_request_error_rate" in unified_config.unified_metrics
        )

        # default config - will not have unified config
        unified_config = get_unified_config(metric="random", namespace="abc")
        self.assertFalse(unified_config)

    def test_get_app_config_time(self):
        _start_time = time.perf_counter()
        app_config = get_app_config(
            metric="rollout_latency", namespace="sandbox_numalogic_demo1"
        )
        time1 = time.perf_counter()-_start_time
        _start_time = time.perf_counter()

        # from given config
        app_config = get_app_config(
            metric="rollout_latency", namespace="sandbox_numalogic_demo1"
        )
        time2 = time.perf_counter()-_start_time

        print(time1)
        print(time2)
        self.assertTrue(time2 < time1)


@patch.object(tools, "get_all_configs", Mock(return_value=mock_configs()))
class TestWindowScorer(unittest.TestCase):
    def test_get_winscore(self):
        metric_conf = get_metric_config(
            metric="namespace_app_rollouts_http_request_error_rate",
            namespace="sandbox_numalogic_demo2",
        )
        stream = np.random.uniform(low=1, high=2, size=(10, 1))
        payload = StreamPayload(
            uuid="123",
            composite_keys=OrderedDict({"namespace": "sandbox_numalogic_demo2"}),
            win_raw_arr=stream,
            win_arr=stream.copy(),
            win_ts_arr=list(range(10)),
        )
        winscorer = WindowScorer(metric_conf)
        final_score = winscorer.get_final_winscore(payload)
        self.assertLess(final_score, 3.0)
