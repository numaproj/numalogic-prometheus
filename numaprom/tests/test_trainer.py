import os
import unittest
from unittest.mock import patch, Mock

from numalogic.registry import MLflowRegistrar

import trainer
from numaprom.constants import TESTS_DIR, METRIC_CONFIG
from numaprom.prometheus import Prometheus
from numaprom.tests.tools import (
    return_mock_metric_config,
    return_mock_lstmae,
    mock_argocd_query_metric,
    mock_rollout_query_metric,
)

DATA_DIR = os.path.join(TESTS_DIR, "resources", "data")
STREAM_DATA_PATH = os.path.join(DATA_DIR, "stream.json")


@patch.dict(METRIC_CONFIG, return_mock_metric_config())
@patch.object(trainer, "save_model", Mock(return_value=1))
class TestTrainer(unittest.TestCase):
    @patch.object(Prometheus, "query_metric", Mock(return_value=mock_argocd_query_metric()))
    def test_argocd_trainer(self):
        _out = trainer.argocd_trainer(
            {"namespace": "sandbox_numalogic_demo", "name": "metric_1", "resume_training": False}
        )
        self.assertEqual(_out, 1)

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

    @patch.object(Prometheus, "query_metric", Mock(return_value=mock_rollout_query_metric()))
    @patch.object(MLflowRegistrar, "load", Mock(return_value=return_mock_lstmae()))
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
