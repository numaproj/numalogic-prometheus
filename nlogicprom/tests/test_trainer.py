import datetime
import os
import unittest
from unittest.mock import patch, Mock

import pandas as pd
import torch
from mlflow.entities.model_registry import ModelVersion
from numalogic.models.autoencoder.variants import LSTMAE
from numalogic.registry import MLflowRegistrar

import trainer
from nlogicprom.constants import TESTS_DIR
from nlogicprom.prometheus import Prometheus
from nlogicprom.tests.tools import MODEL_DIR

DATA_DIR = os.path.join(TESTS_DIR, "resources", "data")
STREAM_DATA_PATH = os.path.join(DATA_DIR, "stream.json")


def mock_query_metric(*_, **__):
    return pd.read_csv(
        os.path.join(TESTS_DIR, "resources", "data", "2xx_w_nan.csv"),
        index_col="timestamp",
        parse_dates=["timestamp"],
        infer_datetime_format=True,
    )


def mock_rollout_query_metric(*_, **__):
    return pd.read_csv(
        os.path.join(TESTS_DIR, "resources", "data", "rollout_latency.csv"),
        index_col="timestamp",
        parse_dates=["timestamp"],
        infer_datetime_format=True,
    )


@patch.object(Prometheus, "query_metric", Mock(return_value=mock_query_metric()))
@patch.object(trainer, "save_model", Mock(return_value=1))
class TestTrain(unittest.TestCase):
    @patch.dict("os.environ", {"WIN_SIZE": "4"})
    def test_train1(self):
        _out = trainer.train('{"namespace":"sandbox-meshdr-usw2-demo", "metric": "2xx"}')
        self.assertEqual(_out, 1)


@patch.object(Prometheus, "query_metric", Mock(return_value=mock_rollout_query_metric()))
@patch.object(trainer, "save_model", Mock(return_value=1))
@patch.object(MLflowRegistrar, "load", Mock(return_value=None))
class TestRolloutTrain(unittest.TestCase):
    @patch.dict("os.environ", {"WIN_SIZE": "8"})
    def test_train1(self):
        _out = trainer.rollout_train(
            '{"namespace":"sandbox-meshdr-usw2-demo", "metric": "hash_latency"}'
        )
        self.assertEqual(_out, 1)


def return_mock_model(rollout_win_size=None, *_, **__):
    return {
        "primary_artifact": LSTMAE(seq_len=12, no_features=1, embedding_dim=64),
        "metadata": torch.load(os.path.join(MODEL_DIR, "model_cpu.pth")),
        "model_properties": ModelVersion(
            creation_timestamp=1656615600000,
            current_stage="Production",
            description="",
            last_updated_timestamp=datetime.datetime.now().timestamp() * 1000,
            name="sandbox:lol::demo:lol",
            run_id="6f1e582fb6194bbdaa4141feb2ce8e27",
            run_link="",
            source="mlflow-artifacts:/0/6f1e582fb6194bbdaa4141feb2ce8e27/artifacts/model",
            status="READY",
            status_message="",
            tags={},
            user_id="",
            version="125",
        ),
    }


@patch.object(Prometheus, "query_metric", Mock(return_value=mock_rollout_query_metric()))
@patch.object(MLflowRegistrar, "load", Mock(return_value=return_mock_model()))
@patch.object(trainer, "save_model", Mock(return_value=1))
class TestRolloutResumeTrain2(unittest.TestCase):
    @patch.dict("os.environ", {"WIN_SIZE": "12"})
    def test_train1(self):
        _out = trainer.rollout_train(
            '{"namespace":"sandbox-meshdr-usw2-demo", "metric": "hash_latency"}'
        )
        self.assertEqual(_out, 1)


if __name__ == "__main__":
    unittest.main()
