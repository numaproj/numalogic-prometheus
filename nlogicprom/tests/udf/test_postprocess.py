import os
import unittest
from unittest.mock import patch

from nlogicprom.constants import TESTS_DIR, ARGOCD_METRICS_LIST
from nlogicprom.entities import PrometheusPayload, Payload
from nlogicprom.tests import *
from nlogicprom.tests.tools import (
    get_postproc_input,
    return_mock_cpu_load,
)
from nlogicprom.udf.postprocess import postprocess, save_to_redis

DATA_DIR = os.path.join(TESTS_DIR, "resources", "data")
MODEL_DIR = os.path.join(TESTS_DIR, "resources", "models")
STREAM_DATA_PATH = os.path.join(DATA_DIR, "stream.json")
ROLLOUTS_STREAM_PATH = os.path.join(DATA_DIR, "rollouts_stream.json")


@patch.dict("os.environ", {"WIN_SIZE": "12"})
class TestPostProcess(unittest.TestCase):
    postproc_input = None
    payload = Payload(
        uuid="1234",
        namespace="sandbox-meshdr-usw2-demo",
        metric="cpu",
        src_labels={},
        inputMetrics=[],
        processedMetrics=[],
        startTS="1654121191689",
        endTS="1654121213989",
        anomaly=1,
    )

    @classmethod
    @patch("numalogic.registry.MLflowRegistrar.load", return_mock_cpu_load)
    def setUpClass(cls) -> None:
        redis_client.flushall()
        cls.postproc_input = get_postproc_input(STREAM_DATA_PATH)
        cls.rollouts_postproc_input = get_postproc_input(ROLLOUTS_STREAM_PATH)

    def test_postprocess(self):
        _out = postprocess(None, self.postproc_input)
        data = _out.items()[0]._value.decode("utf-8")
        payload = PrometheusPayload.from_json(data)
        self.assertEqual(payload.name, "namespace_app_pod_cpu_anomaly")

    def test_postprocess_rollouts(self):
        _out = postprocess(None, self.rollouts_postproc_input)
        data = _out.items()[0]._value.decode("utf-8")
        payload = PrometheusPayload.from_json(data)
        self.assertEqual(payload.name, "namespace_rollout_hash_error_rate_anomaly")
        self.assertEqual(payload.labels["hash_id"], "64f9bb588")

    def test_save_redis1(self):
        _out = save_to_redis(self.payload, recreate=False)
        self.assertEqual(-1, _out[0])

    def test_save_redis2(self):
        redis_client.flushall()
        _out = None
        value = 1
        for m in ARGOCD_METRICS_LIST:
            payload = self.payload
            payload.metric = m
            payload.anomaly = value
            _out = save_to_redis(payload, recreate=False)
            value = value + 1
        self.assertEqual(5, _out[0])

    def test_postprocess_unified(self):
        redis_client.flushall()
        value = 1
        for m in ARGOCD_METRICS_LIST:
            payload = self.payload
            payload.metric = m
            payload.anomaly = value
            if m != "cpu":
                _out = save_to_redis(payload, recreate=False)
                value = value + 1
        _out = postprocess(None, self.postproc_input)
        self.assertEqual(len(_out.items()), 2)
        data = _out.items()[1]._value.decode("utf-8")
        payload = PrometheusPayload.from_json(data)
        self.assertEqual(payload.name, "namespace_app_pod_unified_anomaly")

    def test_postprocess_unified_rollouts(self):
        redis_client.flushall()
        payload = self.payload
        payload.namespace = "sandbox-rollout-demo-app"
        payload.metric = "hash_latency"
        payload.hash_id = "64f9bb588"
        _out = save_to_redis(payload, recreate=False)
        _out = postprocess(None, self.rollouts_postproc_input)
        self.assertEqual(len(_out.items()), 2)
        data = _out.items()[1]._value.decode("utf-8")
        payload = PrometheusPayload.from_json(data)
        self.assertEqual(payload.name, "namespace_rollout_hash_unified_anomaly")
        self.assertEqual(payload.labels["hash_id"], "64f9bb588")


if __name__ == "__main__":
    unittest.main()
