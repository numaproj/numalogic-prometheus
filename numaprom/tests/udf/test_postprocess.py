import os
import unittest
from unittest.mock import patch

from numaprom.constants import TESTS_DIR, METRIC_CONFIG, MODEL_CONFIG
from numaprom.entities import PrometheusPayload, Payload
from numaprom.tests import *
from numaprom.tests.tools import (
    get_postproc_input,
    return_mock_vanilla,
    return_mock_metric_config,
    get_datum,
)
from numaprom.udf.postprocess import postprocess, save_to_redis

DATA_DIR = os.path.join(TESTS_DIR, "resources", "data")
MODEL_DIR = os.path.join(TESTS_DIR, "resources", "models")
STREAM_DATA_PATH = os.path.join(DATA_DIR, "stream.json")


@patch.dict(METRIC_CONFIG, return_mock_metric_config())
class TestPostProcess(unittest.TestCase):
    postproc_input = None
    payload = Payload(
        uuid="1234",
        metric_name="metric_1",
        key_map={},
        src_labels={},
        processedMetrics=[],
        startTS="1654121191689",
        endTS="1654121213989",
        anomaly=1,
    )

    @classmethod
    @patch.dict(METRIC_CONFIG, return_mock_metric_config())
    @patch("numalogic.registry.MLflowRegistrar.load", return_mock_vanilla)
    def setUpClass(cls) -> None:
        redis_client.flushall()
        cls.postproc_input = get_postproc_input(STREAM_DATA_PATH)

    def test_postprocess(self):
        for msg in self.postproc_input.items():
            _in = get_datum(msg.value)
            _out = postprocess("", _in)
            data = _out.items()[0].value.decode("utf-8")
            payload = PrometheusPayload.from_json(data)
            self.assertTrue(payload)
            if payload.name != "metric_3_anomaly":
                self.assertEqual(len(_out.items()), 2)
                data = _out.items()[1].value.decode("utf-8")
                payload = PrometheusPayload.from_json(data)
                self.assertTrue(payload)

    def test_save_redis1(self):
        _out = save_to_redis(self.payload, recreate=False)
        self.assertEqual(1, _out[0])

    def test_save_redis2(self):
        redis_client.flushall()
        _out = None
        value = 1
        print(self.payload)
        for m in MODEL_CONFIG["argo_cd"]["metrics"]:
            payload = self.payload
            payload.metric = m
            payload.anomaly = value
            _out = save_to_redis(payload, recreate=False)
            value = value + 1
        print(_out)
        self.assertEqual(5, _out[0])


if __name__ == "__main__":
    unittest.main()
