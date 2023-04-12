import time
import unittest
from unittest.mock import patch, Mock

from numaprom.watcher import ConfigManager
from tests.tools import mock_configs


@patch.object(ConfigManager, "load_configs", Mock(return_value=mock_configs()))
class TestConfigManager(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cm = ConfigManager()
        cls.payload = {"name": "rollout_latency", "namespace": "sandbox_numalogic_demo1"}
        cls.argocd_payload = {
            "name": "namespace_app_http_server_requests_error_rate",
            "namespace": "abc",
        }
        cls.rollouts_payload = {
            "name": "namespace_app_rollouts_http_request_error_rate",
            "namespace": "abc",
        }
        cls.random_payload = {"name": "random", "namespace": "abc"}

    def test_update_configs(self):
        config = self.cm.update_configs()
        self.assertTrue(len(config), 3)

    def test_load_configs(self):
        app_configs, default_configs, default_numalogic = self.cm.load_configs()
        print(type(app_configs))
        print(type(default_configs))

    def test_get_app_config(self):
        # from given config
        app_config = self.cm.get_app_config(
            metric=self.payload["name"], namespace=self.payload["namespace"]
        )
        self.assertTrue(app_config)
        self.assertEqual(app_config.namespace, "sandbox_numalogic_demo1")

        # from given default config
        app_config = self.cm.get_app_config(
            metric=self.rollouts_payload["name"], namespace=self.rollouts_payload["namespace"]
        )
        self.assertTrue(app_config)
        self.assertEqual(app_config.namespace, "default-argorollouts")

        app_config = self.cm.get_app_config(
            metric=self.argocd_payload["name"], namespace=self.argocd_payload["namespace"]
        )
        self.assertTrue(app_config)
        self.assertEqual(app_config.namespace, "default-argocd")

        # default config
        service_config = self.cm.get_app_config(
            metric=self.random_payload["name"], namespace=self.random_payload["namespace"]
        )
        self.assertTrue(service_config)
        self.assertEqual(service_config.namespace, "default")

    def test_get_metric_config(self):
        # from given config
        metric_config = self.cm.get_metric_config(self.payload)
        self.assertTrue(metric_config)
        self.assertEqual(metric_config.metric, "rollout_latency")

        # from given default config
        metric_config = self.cm.get_metric_config(self.rollouts_payload)
        self.assertTrue(metric_config)
        self.assertEqual(metric_config.metric, "namespace_app_rollouts_http_request_error_rate")

        # default config
        metric_config = self.cm.get_metric_config(self.random_payload)
        self.assertTrue(metric_config)
        self.assertEqual(metric_config.metric, "default")

    def test_get_unified_config(self):
        # from given config
        unified_config = self.cm.get_unified_config(self.payload)
        self.assertTrue(unified_config)
        self.assertTrue("rollout_latency" in unified_config.unified_metrics)

        # from given default config
        unified_config = self.cm.get_unified_config(self.rollouts_payload)
        self.assertTrue(unified_config)
        self.assertTrue(
            "namespace_app_rollouts_http_request_error_rate" in unified_config.unified_metrics
        )

        # default config - will not have unified config
        unified_config = self.cm.get_unified_config(self.random_payload)
        self.assertFalse(unified_config)

    def test_get_app_config_time(self):
        _start_time = time.perf_counter()
        ConfigManager().get_app_config(
            metric=self.payload["name"], namespace=self.payload["namespace"]
        )
        time1 = time.perf_counter() - _start_time
        _start_time = time.perf_counter()
        ConfigManager().get_app_config(
            metric=self.payload["name"], namespace=self.payload["namespace"]
        )
        time2 = time.perf_counter() - _start_time
        _start_time = time.perf_counter()
        self.assertTrue(ConfigManager().get_app_config.cache_info().hits >= 1)
        self.assertTrue(time2 < time1)

    def test_get_metric_config_time(self):
        _start_time = time.perf_counter()
        ConfigManager().get_metric_config(self.payload)
        time1 = time.perf_counter() - _start_time
        _start_time = time.perf_counter()
        ConfigManager().get_metric_config(self.payload)
        time2 = time.perf_counter() - _start_time
        self.assertTrue(time2 < time1)

    def test_get_unified_config_time(self):
        _start_time = time.perf_counter()
        ConfigManager().get_unified_config(self.payload)
        time1 = time.perf_counter() - _start_time
        _start_time = time.perf_counter()
        ConfigManager().get_unified_config(self.payload)
        time2 = time.perf_counter() - _start_time
        self.assertTrue(time2 < time1)
