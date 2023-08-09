import datetime
import json
import os
import sys
from unittest import mock
from unittest.mock import MagicMock, patch, Mock

import fakeredis
import numpy as np
import pandas as pd
from numalogic.models.autoencoder.variants import VanillaAE, LSTMAE
from numalogic.models.threshold import StdDevThreshold
from numalogic.registry import ArtifactData, RedisRegistry
from pynumaflow.function import Datum, Messages
from pynumaflow.function._dtypes import DROP, DatumMetadata
from sklearn.preprocessing import MinMaxScaler

from numaprom._constants import TESTS_DIR, POSTPROC_VTX_KEY
from numaprom.factory import HandlerFactory
from tests import window, preprocess

sys.modules["numaprom.mlflow"] = MagicMock()
MODEL_DIR = os.path.join(TESTS_DIR, "resources", "models")


def mockenv(**envvars):
    return mock.patch.dict(os.environ, envvars, clear=True)


def get_datum(data: str or bytes) -> Datum:
    if not isinstance(data, bytes):
        data = json.dumps(data).encode("utf-8")

    return Datum(
        keys=["random_key"],
        value=data,
        event_time=datetime.datetime.now(),
        watermark=datetime.datetime.now(),
        metadata=DatumMetadata(msg_id="", num_delivered=0),
    )


def get_stream_data(data_path: str) -> dict[str, dict | str | list]:
    with open(data_path) as fp:
        data = json.load(fp)
    return data


def get_mock_redis_client():
    server = fakeredis.FakeServer()
    redis_client = fakeredis.FakeStrictRedis(server=server, decode_responses=False)
    return redis_client


def get_prepoc_input(data_path: str) -> Messages:
    out = Messages()
    data = get_stream_data(data_path)
    for obj in data:
        _out = window([""], get_datum(obj))
        if len(_out[0].tags) > 0:
            if not _out[0].tags[0] == DROP:
                out.append(_out[0])
        else:
            out.append(_out[0])
    return out


def get_inference_input(data_path: str, prev_clf_exists=True) -> Messages:
    out = Messages()
    preproc_input = get_prepoc_input(data_path)
    _mock_return = return_preproc_clf() if prev_clf_exists else None
    with patch.object(RedisRegistry, "load", Mock(return_value=_mock_return)):
        for msg in preproc_input:
            _in = get_datum(msg.value)
            _out = preprocess([""], _in)
            if len(_out[0].tags) > 0:
                if not _out[0].tags[0] == DROP:
                    out.append(_out[0])
            else:
                out.append(_out[0])
    return out


def get_threshold_input(data_path: str, prev_clf_exists=True, prev_model_stale=False) -> Messages:
    out = Messages()
    inference_input = get_inference_input(data_path)
    if prev_model_stale:
        _mock_return = return_stale_model_redis()
    elif prev_clf_exists:
        _mock_return = return_mock_lstmae()
    else:
        _mock_return = None
    with patch.object(RedisRegistry, "load", Mock(return_value=_mock_return)):
        for msg in inference_input:
            _in = get_datum(msg.value)
            handler_ = HandlerFactory.get_handler("inference")
            _out = handler_(None, _in)
            if len(_out[0].tags) > 0:
                if not _out[0].tags[0] == DROP:
                    out.append(_out[0])
            else:
                out.append(_out[0])
    return out


def get_postproc_input(data_path: str, prev_clf_exists=True, prev_model_stale=False) -> Messages:
    out = Messages()
    thresh_input = get_threshold_input(data_path, prev_model_stale=prev_model_stale)
    _mock_return = return_threshold_clf() if prev_clf_exists else None
    with patch.object(RedisRegistry, "load", Mock(return_value=_mock_return)):
        for msg in thresh_input:
            _in = get_datum(msg.value)
            handler_ = HandlerFactory.get_handler("threshold")
            _out = handler_(None, _in)
            for _msg in _out:
                for tag in _msg.tags:
                    if tag == bytes(POSTPROC_VTX_KEY, "utf-8"):
                        out.append(_msg)
    return out


def return_mock_lstmae(*_, **__):
    return ArtifactData(
        artifact=LSTMAE(seq_len=2, no_features=1, embedding_dim=4),
        metadata={},
        extras={
            "creation_timestamp": 1653402941169,
            "timestamp": 1653402941,
            "current_stage": "Production",
            "description": "",
            "last_updated_timestamp": 1645369200000,
            "name": "test::error",
            "run_id": "a7c0b376530b40d7b23e6ce2081c899c",
            "run_link": "",
            "source": "mlflow-artifacts:/0/a7c0b376530b40d7b23e6ce2081c899c/artifacts/model",
            "status": "READY",
            "status_message": "",
            "tags": {},
            "user_id": "",
            "version": "5",
        },
    )


def return_stale_model(*_, **__):
    return ArtifactData(
        artifact=VanillaAE(seq_len=2),
        metadata={},
        extras={
            "creation_timestamp": 1653402941169,
            "timestamp": 1653402941,
            "current_stage": "Production",
            "description": "",
            "last_updated_timestamp": 1656615600000,
            "name": "test::error",
            "run_id": "a7c0b376530b40d7b23e6ce2081c899c",
            "run_link": "",
            "source": "mlflow-artifacts:/0/a7c0b376530b40d7b23e6ce2081c899c/artifacts/model",
            "status": "READY",
            "status_message": "",
            "tags": {},
            "user_id": "",
            "version": "5",
        },
    )


def return_stale_model_redis(*_, **__):
    return ArtifactData(
        artifact=VanillaAE(seq_len=2),
        metadata={},
        extras={
            "creation_timestamp": 1653402941169,
            "timestamp": 1653402941,
            "current_stage": "Production",
            "last_updated_timestamp": 1656615600000,
            "name": "test::error",
            "run_id": "a7c0b376530b40d7b23e6ce2081c899c",
            "run_link": "",
            "source": "registry",
            "version": "5",
        },
    )


def return_preproc_clf(n_feat=1):
    x = np.random.randn(100, n_feat)
    clf = MinMaxScaler()
    clf.fit(x)
    return ArtifactData(
        artifact=clf,
        metadata={},
        extras={
            "creation_timestamp": 1653402941169,
            "current_stage": "Production",
            "description": "",
            "last_updated_timestamp": 1656615600000,
            "name": "test::preproc",
            "run_id": "a7c0b376530b40d7b23e6ce2081c899c",
            "run_link": "",
            "source": "mlflow-artifacts:/0/a7c0b376530b40d7b23e6ce2081c899c/artifacts/preproc",
            "status": "READY",
            "status_message": "",
            "tags": {},
            "user_id": "",
            "version": "1",
        },
    )


def return_threshold_clf(n_feat=1):
    x = np.random.randn(100, n_feat)
    clf = StdDevThreshold()
    clf.fit(x)
    return ArtifactData(
        artifact=clf,
        metadata={},
        extras={
            "creation_timestamp": 1653402941169,
            "current_stage": "Production",
            "description": "",
            "last_updated_timestamp": 1656615600000,
            "name": "test::thresh",
            "run_id": "a7c0b376530b40d7b23e6ce2081c899c",
            "run_link": "",
            "source": "mlflow-artifacts:/0/a7c0b376530b40d7b23e6ce2081c899c/artifacts/thresh",
            "status": "READY",
            "status_message": "",
            "tags": {},
            "user_id": "",
            "version": "1",
        },
    )


def return_threshold_clf_redis(n_feat=1):
    x = np.random.randn(100, n_feat)
    clf = StdDevThreshold()
    clf.fit(x)
    return ArtifactData(
        artifact=clf,
        metadata={},
        extras={
            "creation_timestamp": 1653402941169,
            "current_stage": "Production",
            "description": "",
            "last_updated_timestamp": 1656615600000,
            "name": "test::thresh",
            "run_id": "a7c0b376530b40d7b23e6ce2081c899c",
            "run_link": "",
            "source": "registry",
            "status": "READY",
            "version": "1",
        },
    )


def mock_argocd_query_metric(*_, **__):
    return pd.read_csv(
        os.path.join(TESTS_DIR, "resources", "data", "argocd.csv"),
        index_col=0,
        parse_dates=["timestamp"],
    )


def mock_rollout_query_metric(*_, **__):
    return pd.read_csv(
        os.path.join(TESTS_DIR, "resources", "data", "argorollouts.csv"),
        index_col=0,
        parse_dates=["timestamp"],
    )


def mock_rollout_query_metric3(*_, **__):
    data = pd.read_csv(
        os.path.join(TESTS_DIR, "resources", "data", "argorollouts.csv"),
        index_col=0,
        parse_dates=["timestamp"],
    )
    data["extra_column"] = 1
    return data


def mock_rollout_query_metric2(*_, **__):
    df = pd.read_csv(
        os.path.join(TESTS_DIR, "resources", "data", "argorollouts.csv"),
        index_col=0,
        parse_dates=["timestamp"],
    )
    df.rename(columns={"hash_id": "rollouts_pod_template_hash"}, inplace=True)
    return df
