import json
import os
import time
import uuid
import socket
import logging

import mlflow
import pandas as pd
from functools import wraps
from typing import List, Optional, Any, Dict, Sequence

from mlflow.entities.model_registry import ModelVersion
from numalogic.registry import MLflowRegistrar
from pynumaflow.function import Messages, Message

from numaprom.constants import DEFAULT_TRACKING_URI, METRIC_CONFIG
from numaprom.entities import Payload, Metric, Status

LOGGER = logging.getLogger(__name__)


def catch_exception(func):
    @wraps(func)
    def inner_function(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as ex:
            LOGGER.exception("Error in %s: %r", func.__name__, ex)

    return inner_function


def msgs_forward(handler_func):
    @wraps(handler_func)
    def inner_function(*args, **kwargs):
        json_list = handler_func(*args, **kwargs)
        msgs = Messages()
        for json_data in json_list:
            if json_data:
                msgs.append(Message.to_all(json_data.encode()))
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
            msgs.append(Message.to_all(value=json_data.encode()))
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
                msgs.append(Message.to_vtx(key=vertex.encode(), value=json_data.encode()))
            else:
                msgs.append(Message.to_drop())
        return msgs

    return inner_function


def get_key_map(msg: dict) -> Dict:
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


def extract(data: Dict[str, Any]) -> Optional[Payload]:
    input_metrics = [Metric(**_item) for _item in data["window"]]

    payload = Payload(
        uuid=str(uuid.uuid4()),
        metric_name=data["name"],
        key_map=get_key_map(data),
        src_labels=data["labels"],
        processedMetrics=input_metrics,
        startTS=data["timestamp"],
        endTS=data["timestamp"],
        status=Status.EXTRACTED,
    )
    LOGGER.info(
        "%s - Extracted Payload: Keys=%s, Metrics=%s",
        payload.uuid,
        payload.key_map,
        payload.processedMetrics,
    )
    return payload


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
            LOGGER.warning("Failed to resolve hostname: %s: error: %r", hostname, ex, exc_info=True)
            time.sleep(sleep_sec)
        else:
            return True
    LOGGER.error("Failed to resolve hostname: %s even after retries!")
    return False


def load_model(skeys: Sequence[str], dkeys: Sequence[str]) -> Optional[Dict]:
    try:
        tracking_uri = os.getenv("TRACKING_URI", DEFAULT_TRACKING_URI)
        ml_registry = MLflowRegistrar(tracking_uri=tracking_uri)
        artifact_dict = ml_registry.load(skeys=skeys, dkeys=dkeys)
        return artifact_dict
    except Exception as ex:
        print(ex)
        LOGGER.error("Error while loading model from MLflow database: %s", ex)
        return None


def save_model(
    skeys: Sequence[str], dkeys: Sequence[str], model, **metadata
) -> Optional[ModelVersion]:
    tracking_uri = os.getenv("TRACKING_URI", DEFAULT_TRACKING_URI)
    ml_registry = MLflowRegistrar(tracking_uri=tracking_uri, artifact_type="pytorch")
    mlflow.start_run()
    version = ml_registry.save(skeys=skeys, dkeys=dkeys, primary_artifact=model, **metadata)
    LOGGER.info("Successfully saved the model to mlflow. Model version: %s", version)
    mlflow.end_run()
    return version


def get_metric_config(metric_name: str):
    if metric_name in METRIC_CONFIG:
        return METRIC_CONFIG[metric_name]
    else:
        return METRIC_CONFIG["default"]
