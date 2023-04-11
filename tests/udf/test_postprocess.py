import os
import unittest

import orjson
from freezegun import freeze_time
from unittest.mock import patch, Mock

from numaprom._constants import TESTS_DIR
from numaprom.entities import PrometheusPayload, StreamPayload, Header
from numaprom.watcher import ConfigManager
from tests import redis_client
from tests.tools import (
    get_postproc_input,
    get_datum,
    mock_configs,
)
from numaprom.udf.postprocess import postprocess

DATA_DIR = os.path.join(TESTS_DIR, "resources", "data")
MODEL_DIR = os.path.join(TESTS_DIR, "resources", "models")
STREAM_DATA_PATH = os.path.join(DATA_DIR, "stream.json")


@patch("numaprom.tools.set_aws_session", Mock(return_value=None))
@patch.object(ConfigManager, "load_configs", Mock(return_value=mock_configs()))
class TestPostProcess(unittest.TestCase):
    postproc_input = None

    def setUp(self) -> None:
        redis_client.flushall()

    @freeze_time("2022-02-20 12:00:00")
    def test_postprocess(self):
        postproc_input = get_postproc_input(STREAM_DATA_PATH)
        assert postproc_input.items(), print("input items is empty", postproc_input)
        for msg in postproc_input.items():
            _in = get_datum(msg.value)
            stream_payload = StreamPayload(**orjson.loads(msg.value))
            self.assertEqual(Header.MODEL_INFERENCE, stream_payload.header)

            _out = postprocess("", _in)
            data = _out.items()[0].value.decode("utf-8")
            prom_payload = PrometheusPayload.from_json(data)
            self.assertTrue(prom_payload)
            if len(_out.items()) > 1:
                data = _out.items()[1].value.decode("utf-8")
                unified_payload = PrometheusPayload.from_json(data)
                self.assertTrue(unified_payload)

    def test_preprocess_prev_stale_model(self):
        postproc_input = get_postproc_input(STREAM_DATA_PATH, prev_model_stale=True)
        assert postproc_input.items(), print("input items is empty", postproc_input)

        for msg in postproc_input.items():
            _in = get_datum(msg.value)
            stream_payload = StreamPayload(**orjson.loads(msg.value))
            self.assertEqual(Header.MODEL_STALE, stream_payload.header)

            _out = postprocess("", _in)
            data = _out.items()[0].value.decode("utf-8")
            prom_payload = PrometheusPayload.from_json(data)
            self.assertTrue(prom_payload)
            if len(_out.items()) > 1:
                data = _out.items()[1].value.decode("utf-8")
                unified_payload = PrometheusPayload.from_json(data)
                self.assertTrue(unified_payload)

    def test_preprocess_no_prev_clf(self):
        postproc_input = get_postproc_input(STREAM_DATA_PATH, prev_clf_exists=False)
        assert postproc_input.items(), print("input items is empty", postproc_input)

        for msg in postproc_input.items():
            _in = get_datum(msg.value)
            stream_payload = StreamPayload(**orjson.loads(msg.value))
            self.assertEqual(Header.STATIC_INFERENCE, stream_payload.header)

            _out = postprocess("", _in)
            data = _out.items()[0].value.decode("utf-8")
            prom_payload = PrometheusPayload.from_json(data)
            self.assertEqual(prom_payload.labels["model_version"], "-1")

            if len(_out.items()) > 1:
                data = _out.items()[1].value.decode("utf-8")
                unified_payload = PrometheusPayload.from_json(data)
                self.assertEqual(unified_payload.labels["model_version"], "-1")
                self.assertTrue(unified_payload)


if __name__ == "__main__":
    unittest.main()
