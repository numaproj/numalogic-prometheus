import os
import socket
import unittest
from unittest.mock import patch, Mock

from numaprom import tools
from numaprom._constants import TESTS_DIR
from numaprom.tools import (
    is_host_reachable,
    get_metric_config,
    get_service_config,
    get_unified_config,
)
from tests.tools import mock_configs

DATA_DIR = os.path.join(TESTS_DIR, "resources", "data")
REQ_2xx = os.path.join(DATA_DIR, "2xx.txt")
STREAM_DATA_PATH = os.path.join(DATA_DIR, "stream.json")


def mock_resolver(*_, **__):
    raise socket.gaierror


@patch.object(tools, "get_all_configs", Mock(return_value=mock_configs()))
class TestTools(unittest.TestCase):
    INFER_OUT = None

    def test_is_host_reachable(self):
        self.assertTrue(is_host_reachable("google.com"))

    @patch("numaprom.tools.get_ipv4_by_hostname", mock_resolver)
    def test_is_host_reachable_err(self):
        self.assertFalse(is_host_reachable("google.com", max_retries=2, sleep_sec=1))

    def test_get_metric_config(self):
        # from given config
        metric_config = get_metric_config(
            metric="rollout_latency", namespace="sandbox_numalogic_demo1"
        )
        self.assertTrue(metric_config)
        self.assertEqual(metric_config.metric, "rollout_latency")

        # from given default config
        metric_config = get_metric_config(
            metric="namespace_app_argo_rollouts_error_rate", namespace="abc"
        )
        self.assertTrue(metric_config)
        self.assertEqual(metric_config.metric, "namespace_app_argo_rollouts_error_rate")

        # default config
        metric_config = get_metric_config(metric="random", namespace="abc")
        self.assertTrue(metric_config)
        self.assertEqual(metric_config.metric, "default")

    def test_get_service_config(self):
        # from given config
        service_config = get_service_config(
            metric="rollout_latency", namespace="sandbox_numalogic_demo1"
        )
        self.assertTrue(service_config)
        self.assertEqual(service_config.namespace, "sandbox_numalogic_demo1")

        # from given default config
        service_config = get_service_config(
            metric="namespace_app_argo_rollouts_error_rate", namespace="abc"
        )
        self.assertTrue(service_config)
        self.assertEqual(service_config.namespace, "default-argorollouts")
        service_config = get_service_config(
            metric="namespace_app_argo_cd_error_rate", namespace="abc"
        )
        self.assertTrue(service_config)
        self.assertEqual(service_config.namespace, "default-argocd")

        # default config
        service_config = get_service_config(metric="random", namespace="abc")
        self.assertTrue(service_config)
        self.assertEqual(service_config.namespace, "default")

    def test_get_unified_config(self):
        # from given config
        unified_config = get_unified_config(
            metric="rollout_latency", namespace="sandbox_numalogic_demo1"
        )
        self.assertTrue(unified_config)
        self.assertTrue("rollout_latency" in unified_config.unified_metrics)

        # from given default config
        unified_config = get_unified_config(
            metric="namespace_app_argo_rollouts_error_rate", namespace="abc"
        )
        self.assertTrue(unified_config)
        self.assertTrue("namespace_app_argo_rollouts_error_rate" in unified_config.unified_metrics)

        # default config - will not have unified config
        unified_config = get_unified_config(metric="random", namespace="abc")
        self.assertFalse(unified_config)
