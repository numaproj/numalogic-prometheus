import os
import socket
import time
from collections import OrderedDict
from datetime import timedelta, datetime
from functools import wraps
from json import JSONDecodeError
from typing import Optional, Sequence, List

import pandas as pd
import pytz
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import RestException
from numalogic.config import NumalogicConf
from numalogic.models.threshold import StaticThreshold
from numalogic.registry import MLflowRegistry, ArtifactData
from omegaconf import OmegaConf
from orjson import orjson
from pynumaflow.function import Messages, Message

from numaprom import get_logger
from numaprom._constants import DEFAULT_TRACKING_URI, DEFAULT_PROMETHEUS_SERVER, CONFIG_DIR
from numaprom.config._config import MetricConf, NamespaceConf, NumapromConf, UnifiedConf
from numaprom.entities import TrainerPayload, Status, Header, StreamPayload
from numaprom.prometheus import Prometheus

_LOGGER = get_logger(__name__)


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


def create_composite_keys(msg: dict, keys: List[str]) -> OrderedDict:
    labels = msg.get("labels")
    result = OrderedDict()
    for k in keys:
        if k in msg:
            result[k] = msg[k]
        if k in labels:
            result[k] = labels[k]
    return result


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


def load_model(
    skeys: Sequence[str], dkeys: Sequence[str], artifact_type: str = "pytorch"
) -> Optional[ArtifactData]:
    try:
        tracking_uri = os.getenv("TRACKING_URI", DEFAULT_TRACKING_URI)
        ml_registry = MLflowRegistry(tracking_uri=tracking_uri, artifact_type=artifact_type)
        return ml_registry.load(skeys=skeys, dkeys=dkeys)
    except RestException as warn:
        if warn.error_code == 404:
            return None
        _LOGGER.warning("Non 404 error from mlflow: %r", warn)
    except Exception as ex:
        _LOGGER.error("Unexpected error while loading model from MLflow database: %r", ex)
        return None


def save_model(
    skeys: Sequence[str], dkeys: Sequence[str], model, artifact_type="pytorch", **metadata
) -> Optional[ModelVersion]:
    tracking_uri = os.getenv("TRACKING_URI", DEFAULT_TRACKING_URI)
    ml_registry = MLflowRegistry(tracking_uri=tracking_uri, artifact_type=artifact_type)
    version = ml_registry.save(skeys=skeys, dkeys=dkeys, artifact=model, **metadata)
    return version


def get_all_configs():
    schema: NumapromConf = OmegaConf.structured(NumapromConf)

    conf = OmegaConf.load(os.path.join(CONFIG_DIR, "config.yaml"))
    given_configs = OmegaConf.merge(schema, conf).configs

    conf = OmegaConf.load(os.path.join(CONFIG_DIR, "default", "config.yaml"))
    default_configs = OmegaConf.merge(schema, conf).configs

    conf = OmegaConf.load(os.path.join(CONFIG_DIR, "default", "numalogic.yaml"))
    schema: NumalogicConf = OmegaConf.structured(NumalogicConf)
    default_numalogic = OmegaConf.merge(schema, conf)

    return given_configs, default_configs, default_numalogic


def get_namespace_config(metric: str, namespace: str):
    given_configs, default_configs, default_numalogic = get_all_configs()

    # search and load from given configs
    namespace_config = list(filter(lambda conf: (conf.namespace == namespace), given_configs))

    # if not search and load from default configs
    if not namespace_config:
        for _conf in default_configs:
            if metric in _conf.unified_configs[0].unified_metrics:
                namespace_config = [_conf]
                break

    # if not in default configs, initialize Namespace conf with default values
    if not namespace_config:
        namespace_config = OmegaConf.structured(NamespaceConf)
    else:
        namespace_config = namespace_config[0]

    # loading and setting default numalogic config
    for metric_config in namespace_config.metric_configs:
        if OmegaConf.is_missing(metric_config, "numalogic_conf"):
            metric_config.numalogic_conf = default_numalogic

    return namespace_config


def get_metric_config(metric: str, namespace: str) -> Optional[MetricConf]:
    namespace_config = get_namespace_config(metric, namespace)
    metric_config = list(
        filter(lambda conf: (conf.metric == metric), namespace_config.metric_configs)
    )
    if not metric_config:
        return namespace_config.metric_configs[0]
    return metric_config[0]


def get_unified_config(metric: str, namespace: str) -> Optional[UnifiedConf]:
    namespace_config = get_namespace_config(metric, namespace)
    unified_config = list(
        filter(lambda conf: (metric in conf.unified_metrics), namespace_config.unified_configs)
    )
    if not unified_config:
        return None
    return unified_config[0]


def fetch_data(
    payload: TrainerPayload, metric_config: MetricConf, labels: dict, return_labels=None
) -> pd.DataFrame:
    _start_time = time.time()

    prometheus_server = os.getenv("PROMETHEUS_SERVER", DEFAULT_PROMETHEUS_SERVER)
    datafetcher = Prometheus(prometheus_server)

    end_dt = datetime.now(pytz.utc)
    start_dt = end_dt - timedelta(hours=36)

    df = datafetcher.query_metric(
        metric_name=payload.composite_keys["name"],
        labels_map=labels,
        return_labels=return_labels,
        start=start_dt.timestamp(),
        end=end_dt.timestamp(),
        step=metric_config.scrape_interval,
    )
    _LOGGER.info(
        "%s - Time taken to fetch data: %s, for df shape: %s",
        payload.uuid,
        time.time() - _start_time,
        df.shape,
    )
    return df


def calculate_static_thresh(payload: StreamPayload, upper_limit: float) -> bytes:
    """
    Calculates static thresholding, and returns a serialized json bytes payload.
    """
    x = payload.get_stream_array()
    static_clf = StaticThreshold(upper_limit=upper_limit)
    static_scores = static_clf.score_samples(x)

    payload.set_win_arr(static_scores)
    payload.set_header(Header.STATIC_INFERENCE)
    payload.set_status(Status.ARTIFACT_NOT_FOUND)
    payload.set_metadata("version", -1)

    _LOGGER.info("%s - Static thresholding complete for payload: %s", payload.uuid, payload)
    return orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)
