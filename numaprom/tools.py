import json
import logging
import os
import socket
import time
from datetime import timedelta, datetime
from functools import wraps
from json import JSONDecodeError
from typing import List, Optional, Any, Dict, Sequence

import pandas as pd
import pytz
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import RestException
from numalogic.registry import MLflowRegistry, ArtifactData
from pynumaflow.function import Messages, Message

from numaprom._constants import DEFAULT_TRACKING_URI, METRIC_CONFIG, DEFAULT_PROMETHEUS_SERVER
from numaprom.entities import Metric
from numaprom.prometheus import Prometheus

_LOGGER = logging.getLogger(__name__)


def catch_exception(func):
    @wraps(func)
    def inner_function(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except JSONDecodeError as err:
            _LOGGER.exception("Error in json decode for %s: %r", func.__name__, err)
        except Exception as ex:
            _LOGGER.exception("Error in %s: %r", func.__name__, ex)

    return inner_function


def msgs_forward(handler_func):
    @wraps(handler_func)
    def inner_function(*args, **kwargs):
        json_list = handler_func(*args, **kwargs)
        msgs = Messages()
        for json_data in json_list:
            if json_data:
                msgs.append(Message.to_all(json_data))
            else:
                msgs.append(Message.to_drop())
        return msgs

    return inner_function


def msg_forward(handler_func):
    @wraps(handler_func)
    def inner_function(*args, **kwargs):
        json_data = handler_func(*args, **kwargs)
        msgs = Messages()
        if json_data:
            msgs.append(Message.to_all(value=json_data))
        else:
            msgs.append(Message.to_drop())
        return msgs

    return inner_function


def conditional_forward(hand_func):
    @wraps(hand_func)
    def inner_function(*args, **kwargs) -> Messages:
        data = hand_func(*args, **kwargs)
        msgs = Messages()
        for vertex, json_data in data:
            if json_data and vertex:
                msgs.append(Message.to_vtx(key=vertex.encode(), value=json_data))
            else:
                msgs.append(Message.to_drop())
        return msgs

    return inner_function


def create_composite_keys(msg: dict) -> Dict:
    labels = msg.get("labels")
    metric_name = msg["name"]

    keys = get_metric_config(metric_name)["keys"]
    result = {}
    for k in keys:
        if k in msg:
            result[k] = msg[k]
        if k in labels:
            result[k] = labels[k]
    return result


def get_metrics(df: pd.DataFrame) -> List[Metric]:
    metrics = [Metric(**kwargs) for kwargs in df.to_dict(orient="records")]
    return metrics


def get_data(file_path: str) -> Any:
    file = open(file_path)
    data = json.load(file)
    file.close()
    return data


def get_ipv4_by_hostname(hostname: str, port=0) -> list:
    return list(
        idx[4][0]
        for idx in socket.getaddrinfo(hostname, port)
        if idx[0] is socket.AddressFamily.AF_INET and idx[1] is socket.SocketKind.SOCK_RAW
    )


def is_host_reachable(hostname: str, port=None, max_retries=5, sleep_sec=5) -> bool:
    retries = 0
    assert max_retries >= 1, "Max retries has to be at least 1"

    while retries < max_retries:
        try:
            get_ipv4_by_hostname(hostname, port)
        except socket.gaierror as ex:
            retries += 1
            _LOGGER.warning(
                "Failed to resolve hostname: %s: error: %r", hostname, ex, exc_info=True
            )
            time.sleep(sleep_sec)
        else:
            return True
    _LOGGER.error("Failed to resolve hostname: %s even after retries!")
    return False


def load_model(skeys: Sequence[str], dkeys: Sequence[str]) -> Optional[ArtifactData]:
    try:
        tracking_uri = os.getenv("TRACKING_URI", DEFAULT_TRACKING_URI)
        ml_registry = MLflowRegistry(tracking_uri=tracking_uri)
        return ml_registry.load(skeys=skeys, dkeys=dkeys)
    except RestException as warn:
        if warn.error_code == 404:
            return None
        _LOGGER.warning("Non 404 error from mlflow: %r", warn)
    except Exception as ex:
        _LOGGER.error("Unexpected error while loading model from MLflow database: %r", ex)
        return None


def save_model(
    skeys: Sequence[str], dkeys: Sequence[str], model, **metadata
) -> Optional[ModelVersion]:
    tracking_uri = os.getenv("TRACKING_URI", DEFAULT_TRACKING_URI)
    ml_registry = MLflowRegistry(tracking_uri=tracking_uri, artifact_type="pytorch")
    version = ml_registry.save(skeys=skeys, dkeys=dkeys, artifact=model, **metadata)
    return version


def get_metric_config(metric_name: str):
    if metric_name in METRIC_CONFIG:
        return METRIC_CONFIG[metric_name]
    return METRIC_CONFIG["default"]


def fetch_data(
    uuid: str, metric_name: str, model_config: dict, labels: dict, return_labels=None
) -> pd.DataFrame:
    _start_time = time.time()

    prometheus_server = os.getenv("PROMETHEUS_SERVER", DEFAULT_PROMETHEUS_SERVER)
    datafetcher = Prometheus(prometheus_server)

    end_dt = datetime.now(pytz.utc)
    start_dt = end_dt - timedelta(hours=15)

    df = datafetcher.query_metric(
        metric_name=metric_name,
        labels_map=labels,
        return_labels=return_labels,
        start=start_dt.timestamp(),
        end=end_dt.timestamp(),
        step=model_config["scrape_interval"],
    )
    _LOGGER.info(
        "%s - Time taken to fetch data: %s, for df shape: %s",
        uuid,
        time.time() - _start_time,
        df.shape,
    )
    return df
