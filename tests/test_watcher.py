import unittest
from unittest.mock import patch, Mock

from numaprom import tools

from numaprom.watcher import get_metric_config, get_app_config, get_unified_config, Watcher, ConfigHandler

from tests.tools import mock_configs
from tests import redis_client


@patch.object(tools, "get_all_configs", Mock(return_value=mock_configs()))
class TestTools(unittest.TestCase):

    def tearDown(self) -> None:
        redis_client.flushall()

    def test_watcher(self):
        config_paths = ["./resources/configs", "./numaprom/default-configs"]
        w = Watcher(config_paths, ConfigHandler())
        w.run()

    # def test_get_metric_config(self):
    #     # from given config
    #     metric_config = get_metric_config(
    #         metric="rollout_latency", namespace="sandbox_numalogic_demo1"
    #     )
    #     self.assertTrue(metric_config)
    #     self.assertEqual(metric_config.metric, "rollout_latency")
    #
    #     # from given default config
    #     metric_config = get_metric_config(
    #         metric="namespace_app_rollouts_http_request_error_rate", namespace="abc"
    #     )
    #     self.assertTrue(metric_config)
    #     self.assertEqual(metric_config.metric, "namespace_app_rollouts_http_request_error_rate")
    #
    #     # default config
    #     metric_config = get_metric_config(metric="random", namespace="abc")
    #     self.assertTrue(metric_config)
    #     self.assertEqual(metric_config.metric, "default")
    #
    # def test_get_app_config(self):
    #     # from given config
    #     app_config = get_app_config(
    #         metric="rollout_latency", namespace="sandbox_numalogic_demo1"
    #     )
    #     self.assertTrue(app_config)
    #     self.assertEqual(app_config.namespace, "sandbox_numalogic_demo1")
    #
    #     # from given default config
    #     app_config = get_app_config(
    #         metric="namespace_app_rollouts_http_request_error_rate", namespace="abc"
    #     )
    #     self.assertTrue(app_config)
    #     self.assertEqual(app_config.namespace, "default-argorollouts")
    #     app_config = get_app_config(
    #         metric="namespace_app_http_server_requests_error_rate", namespace="abc"
    #     )
    #     self.assertTrue(app_config)
    #     self.assertEqual(app_config.namespace, "default-argocd")
    #
    #     # default config
    #     app_config = get_app_config(metric="random", namespace="abc")
    #     self.assertTrue(app_config)
    #     self.assertEqual(app_config.namespace, "default")
    #
    # def test_get_unified_config(self):
    #     # from given config
    #     unified_config = get_unified_config(
    #         metric="rollout_latency", namespace="sandbox_numalogic_demo1"
    #     )
    #     self.assertTrue(unified_config)
    #     self.assertTrue("rollout_latency" in unified_config.unified_metrics)
    #
    #     # from given default config
    #     unified_config = get_unified_config(
    #         metric="namespace_app_rollouts_http_request_error_rate", namespace="abc"
    #     )
    #     self.assertTrue(unified_config)
    #     self.assertTrue(
    #         "namespace_app_rollouts_http_request_error_rate" in unified_config.unified_metrics
    #     )
    #
    #     # default config - will not have unified config
    #     unified_config = get_unified_config(metric="random", namespace="abc")
    #     self.assertFalse(unified_config)
