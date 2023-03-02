import os
import unittest
from freezegun import freeze_time
from collections import OrderedDict
from unittest.mock import patch, Mock

from numaprom import tools
from numaprom._constants import TESTS_DIR, OUTPUT_CONFIG
from numaprom.entities import PrometheusPayload, StreamPayload
from numaprom.tools import get_unified_config
from tests import redis_client
from tests.tools import (
    get_postproc_input,
    get_datum, mock_configs,
    mock_numalogic_conf,
)
from numaprom.udf.postprocess import postprocess, save_to_redis

DATA_DIR = os.path.join(TESTS_DIR, "resources", "data")
MODEL_DIR = os.path.join(TESTS_DIR, "resources", "models")
STREAM_DATA_PATH = os.path.join(DATA_DIR, "stream.json")


@patch.object(tools, "get_configs", Mock(return_value=mock_configs()))
@patch.object(tools, "default_numalogic_conf", Mock(return_value=mock_numalogic_conf()))
class TestPostProcess(unittest.TestCase):
    postproc_input = None

    @classmethod
    @freeze_time("2022-02-20 12:00:00")
    def setUpClass(cls) -> None:
        redis_client.flushall()

    def test_postprocess(self):
        postproc_input = get_postproc_input(STREAM_DATA_PATH)
        postproc_input.items(), print("input items is empty", postproc_input)
        for msg in postproc_input.items():
            _in = get_datum(msg.value)
            _out = postprocess("", _in)
            data = _out.items()[0].value.decode("utf-8")
            prom_payload = PrometheusPayload.from_json(data)
            print(prom_payload.name, len(_out.items()))
            # if prom_payload.name != "metric_3_anomaly":
            #     self.assertEqual(len(_out.items()), 2)
            #     data = _out.items()[1].value.decode("utf-8")
            #     unified_payload = PrometheusPayload.from_json(data)
            #     self.assertTrue(unified_payload)

    def test_preprocess_prev_stale_model(self):
        postproc_input = get_postproc_input(STREAM_DATA_PATH, prev_model_stale=True)
        assert postproc_input.items(), print("input items is empty", postproc_input)

        for msg in postproc_input.items():
            _in = get_datum(msg.value)
            _out = postprocess("", _in)
            data = _out.items()[0].value.decode("utf-8")
            prom_payload = PrometheusPayload.from_json(data)

            if prom_payload.name != "metric_3_anomaly":
                self.assertEqual(len(_out.items()), 2)
                data = _out.items()[1].value.decode("utf-8")
                unified_payload = PrometheusPayload.from_json(data)
                self.assertTrue(unified_payload)

    def test_preprocess_no_prev_clf(self):
        postproc_input = get_postproc_input(STREAM_DATA_PATH, prev_clf_exists=False)
        assert postproc_input.items(), print("input items is empty", postproc_input)

        for msg in postproc_input.items():
            _in = get_datum(msg.value)
            _out = postprocess("", _in)
            data = _out.items()[0].value.decode("utf-8")
            prom_payload = PrometheusPayload.from_json(data)
            self.assertEqual(prom_payload.labels["model_version"], "-1")

            if prom_payload.name != "metric_3_anomaly":
                self.assertEqual(len(_out.items()), 2)
                data = _out.items()[1].value.decode("utf-8")
                unified_payload = PrometheusPayload.from_json(data)
                self.assertEqual(unified_payload.labels["model_version"], "-1")
                self.assertTrue(unified_payload)


if __name__ == "__main__":
    unittest.main()
