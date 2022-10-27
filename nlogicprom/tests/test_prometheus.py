import os
import requests
import datetime
import unittest
from unittest.mock import patch, Mock, MagicMock

from nlogicprom.constants import TESTS_DIR
from nlogicprom.prometheus import Prometheus

DATA_DIR = os.path.join(TESTS_DIR, "resources", "data")
STREAM_DATA_PATH = os.path.join(DATA_DIR, "stream.json")


def mock_query_range(*_, **__):
    result = [
        {
            "metric": {
                "__name__": "namespace_asset_pod_cpu_utilization",
                "assetAlias": "sandbox.sandbox.meshdr",
                "numalogic": "true",
                "namespace": "sandbox-meshdr-usw2-demo",
            },
            "values": [[1656334767.73, "14.744611739611193"], [1656334797.73, "14.73040822323633"]],
        }
    ]
    return result


def mock_response(*_, **__):
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "status": "success",
        "data": {
            "resultType": "vector",
            "result": [
                {
                    "metric": {
                        "__name__": "namespace_asset_pod_cpu_utilization",
                        "numalogic": "true",
                        "namespace": "sandbox-meshdr-usw2-demo",
                    },
                    "value": [1666730247.616, "0.7627805793186443"],
                }
            ],
        },
    }
    return response


class TestPrometheus(unittest.TestCase):
    start = None
    end = None
    prom = None

    @classmethod
    def setUpClass(cls) -> None:
        end = datetime.datetime.now()
        start = end - datetime.timedelta(hours=36)
        cls.start = start.timestamp()
        cls.end = end.timestamp()
        cls.prom = Prometheus(prometheus_server="http://localhost:8490")

    @patch.object(Prometheus, "query_range", Mock(return_value=mock_query_range()))
    def test_train1(self):
        _out = self.prom.query_metric(
            namespace="sandbox-meshdr-usw2-demo",
            metric="cpu",
            start=self.start,
            end=self.end,
        )
        self.assertEqual(_out.shape, (2, 1))

    @patch.object(Prometheus, "query_range", Mock(return_value=mock_query_range()))
    def test_train_hash_metric(self):
        _out = self.prom.query_metric(
            namespace="sandbox-meshdr-usw2-demo",
            metric="hash_error_rate",
            start=self.start,
            end=self.end,
        )
        self.assertEqual(_out.shape, (2, 1))

    @patch.object(requests, "get", Mock(return_value=mock_response()))
    def test_query_range(self):
        _out = self.prom.query_range(
            query="namespace_asset_pod_cpu_utilization{" "namespace='sandbox-meshdr-usw2-demo'}",
            start=self.start,
            end=self.end,
        )
        self.assertEqual(len(_out), 1)

    @patch.object(requests, "get", Mock(return_value=mock_response()))
    def test_query(self):
        _out = self.prom.query(
            query="namespace_asset_pod_cpu_utilization{" "namespace='sandbox-meshdr-usw2-demo'}"
        )
        self.assertEqual(len(_out), 1)


if __name__ == "__main__":
    unittest.main()
