import json
import os
import unittest
from datetime import datetime
from typing import Union
from unittest.mock import patch, Mock

from numalogic.registry import MLflowRegistry
from pynumaflow.sink import Datum

import trainer
from numaprom.udsink import train

from numaprom._constants import TESTS_DIR, METRIC_CONFIG
from numaprom.prometheus import Prometheus
from tests.tools import (
    return_mock_metric_config,
    return_mock_lstmae,
    mock_argocd_query_metric,
    mock_rollout_query_metric,
)

DATA_DIR = os.path.join(TESTS_DIR, "resources", "data")
STREAM_DATA_PATH = os.path.join(DATA_DIR, "stream.json")


def as_datum(data: Union[str, bytes, dict]) -> Datum:
    if type(data) is not bytes:
        data = json.dumps(data).encode("utf-8")
    elif type(data) == dict:
        data = json.dumps(data)

    return Datum(sink_msg_id="1", value=data, event_time=datetime.now(), watermark=datetime.now())


@patch.dict(METRIC_CONFIG, return_mock_metric_config())
class TestTrainer(unittest.TestCase):
    train_payload = {
        "namespace": "sandbox_numalogic_demo",
        "name": "metric_1",
        "resume_training": False,
    }

    @patch.object(MLflowRegistry, "save", Mock(return_value=1))
    @patch.object(Prometheus, "query_metric", Mock(return_value=mock_argocd_query_metric()))
    def test_argocd_trainer(self):
        datums = [as_datum(self.train_payload)]
        _out = train(datums)
        self.assertTrue(_out.items()[0].success)
        self.assertEqual("1", _out.items()[0].id)

    @unittest.skip("Need to update for rollouts")
    @patch.object(Prometheus, "query_metric", Mock(return_value=mock_rollout_query_metric()))
    def test_rollout_trainer(self):
        _out = trainer.rollout_trainer(
            {
                "namespace": "sandbox_numalogic_demo",
                "name": "metric_2",
                "hash_id": "123456789",
                "resume_training": False,
            }
        )
        self.assertEqual(_out, 1)

    @unittest.skip("Need to update for rollouts")
    @patch.object(Prometheus, "query_metric", Mock(return_value=mock_rollout_query_metric()))
    @patch.object(MLflowRegistry, "load", Mock(return_value=return_mock_lstmae()))
    def test_rollout_trainer2(self):
        _out = trainer.rollout_trainer(
            {
                "namespace": "sandbox_numalogic_demo",
                "name": "metric_2",
                "hash_id": "123456789",
                "resume_training": True,
            }
        )
        self.assertEqual(_out, 1)


if __name__ == "__main__":
    unittest.main()
