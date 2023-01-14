import os
import unittest
from unittest.mock import patch

from freezegun import freeze_time

from numaprom._constants import TESTS_DIR, METRIC_CONFIG, MODEL_CONFIG
from numaprom.entities import PrometheusPayload, StreamPayload
from tests import redis_client
from tests.tools import (
    get_postproc_input,
    return_mock_metric_config,
    get_datum,
    return_mock_lstmae,
)
from numaprom.udf.postprocess import postprocess, save_to_redis

DATA_DIR = os.path.join(TESTS_DIR, "resources", "data")
MODEL_DIR = os.path.join(TESTS_DIR, "resources", "models")
STREAM_DATA_PATH = os.path.join(DATA_DIR, "stream.json")


@patch.dict(METRIC_CONFIG, return_mock_metric_config())
class TestPostProcess(unittest.TestCase):
    postproc_input = None

    stream_payload = StreamPayload(
        uuid="1234",
        win_arr=[[3.2123, 5.32132]],
        win_ts_arr=["1654121191689", "1654121213989"],
        composite_keys={
            "name": "metric_1",
        },
    )

    @classmethod
    @freeze_time("2022-02-20 12:00:00")
    @patch.dict(METRIC_CONFIG, return_mock_metric_config())
    @patch("numalogic.registry.MLflowRegistry.load", return_mock_lstmae)
    def setUpClass(cls) -> None:
        redis_client.flushall()
        cls.postproc_input = get_postproc_input(STREAM_DATA_PATH)

    def setUp(self) -> None:
        redis_client.flushall()

    def test_postprocess(self):
        for msg in self.postproc_input.items():
            _in = get_datum(msg.value)
            _out = postprocess("", _in)
            data = _out.items()[0].value.decode("utf-8")
            print("DATA", data)
            prom_payload = PrometheusPayload.from_json(data)

            if prom_payload.name != "metric_3_anomaly":
                self.assertEqual(len(_out.items()), 2)
                data = _out.items()[1].value.decode("utf-8")
                uniprom_payload = PrometheusPayload.from_json(data)
                self.assertTrue(uniprom_payload)

    def test_save_redis1(self):
        score = 5.0
        _out = save_to_redis(self.stream_payload, score, recreate=False)
        self.assertEqual(score, _out[0])

    def test_save_redis2(self):
        max_score, scores = None, []
        score = 1

        for m in MODEL_CONFIG["argo_cd"]["metrics"]:
            payload = self.stream_payload
            payload.composite_keys = {"name": m}
            max_score, scores = save_to_redis(payload, score, recreate=False)
            score += 1
        self.assertEqual(5, max_score)
        self.assertListEqual([1.0, 2.0, 3.0, 4.0, 5.0], scores)


if __name__ == "__main__":
    unittest.main()
