import json
import time
import uuid
import socket
import logging
import pandas as pd
from functools import wraps
from typing import List, Optional, Any, Dict

from pynumaflow.function import Messages, Message

from nlogicprom.entities import Payload, Metric, MetricType, Status

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


def get_metric_type(metric: str) -> MetricType:
    if metric == "namespace_app_pod_http_server_requests_errors":
        return MetricType.ERROR_COUNT
    if metric == "namespace_app_pod_http_server_requests_error_rate":
        return MetricType.ERROR_RATE
    elif metric == "namespace_app_pod_http_server_requests_latency":
        return MetricType.LATENCY
    elif metric == "namespace_asset_pod_cpu_utilization":
        return MetricType.CPU
    elif metric == "namespace_asset_pod_memory_utilization":
        return MetricType.MEMORY
    elif metric == "namespace_hash_pod_http_server_requests_error_rate":
        return MetricType.HASH_ERROR_RATE
    elif metric == "namespace_hash_pod_http_server_requests_latency":
        return MetricType.HASH_LATENCY
    else:
        raise NotImplementedError(f"Unsupported metric type: {metric}")


def extract(data: Dict[str, Any]) -> Optional[Payload]:
    input_metrics = [Metric(**_item) for _item in data["window"]]
    if "pod_template_hash" in data["labels"]:
        hash_id = str(data["labels"]["pod_template_hash"])
    elif "rollouts_pod_template_hash" in data["labels"]:
        hash_id = str(data["labels"]["rollouts_pod_template_hash"])
    else:
        hash_id = ""

    payload = Payload(
        uuid=str(uuid.uuid4()),
        namespace=data["labels"]["namespace"],
        metric=get_metric_type(data["name"]).value,
        hash_id=hash_id,
        src_labels=data["labels"],
        inputMetrics=input_metrics,
        processedMetrics=input_metrics,
        startTS=data["timestamp"],
        endTS=data["timestamp"],
        status=Status.EXTRACTED,
    )
    LOGGER.info(
        "%s - Extracted Payload: keys: [%s, %s, %s]",
        payload.uuid,
        payload.namespace,
        payload.metric,
        payload.hash_id,
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


def decode_msg(msg: bytes) -> Dict[str, Any]:
    msg = msg.decode("utf-8")

    try:
        data = json.loads(msg)
    except Exception as ex:
        LOGGER.exception("Error in Json serialization: %r", ex)
        return ""
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
