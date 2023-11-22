import socket
import time
from collections import OrderedDict
from datetime import timedelta, datetime
from functools import wraps
from json import JSONDecodeError

import numpy as np
import pandas as pd
import pytz
from numalogic.config import PostprocessFactory
from numalogic.models.threshold import SigmoidThreshold
from pynumaflow.mapper import Messages, Message
from numaprom import LOGGER, MetricConf
from numaprom.clients.prometheus import Prometheus
from numaprom.entities import TrainerPayload, StreamPayload
from numaprom.watcher import ConfigManager


def catch_exception(func):
    @wraps(func)
    def inner_function(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except JSONDecodeError as err:
            LOGGER.exception("Error in json decode for {name}: {err}", name=func.__name__, err=err)
        except Exception as ex:
            LOGGER.exception("Error in {name}: {err}", name=func.__name__, err=ex)

    return inner_function


def msgs_forward(handler_func):
    @wraps(handler_func)
    def inner_function(*args, **kwargs):
        json_list = handler_func(*args, **kwargs)
        msgs = Messages()
        for json_data in json_list:
            if json_data:
                msgs.append(Message(json_data))
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
            msgs.append(Message(value=json_data))
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
                msgs.append(Message(value=json_data, tags=[vertex.encode()]))
            else:
                msgs.append(Message.to_drop())
        return msgs

    return inner_function


def create_composite_keys(msg: dict, keys: list[str]) -> OrderedDict:
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
            LOGGER.warning(
                "Failed to resolve hostname: {hostname}: error: {ex}",
                hostname=hostname,
                ex=ex,
                exc_info=True,
            )
            time.sleep(sleep_sec)
        else:
            return True
    LOGGER.error("Failed to resolve hostname: {retries} even after retries!", retries=retries)
    return False


def fetch_data(
    payload: TrainerPayload,
    metric_config: MetricConf,
    labels: dict,
    return_labels=None,
    hours: int = 36,
) -> pd.DataFrame:
    _start_time = time.time()
    prometheus_conf = ConfigManager.get_prometheus_config()
    datafetcher = Prometheus(prometheus_conf.server)

    end_dt = datetime.now(pytz.utc)
    start_dt = end_dt - timedelta(hours=hours)

    df = datafetcher.query_metric(
        metric_name=payload.composite_keys["name"],
        labels_map=labels,
        return_labels=return_labels,
        start=start_dt.timestamp(),
        end=end_dt.timestamp(),
        step=metric_config.scrape_interval,
    )
    LOGGER.info(
        "{uuid} - Time taken to fetch data: {time}, for df shape: {shape}",
        uuid=payload.uuid,
        time=time.time() - _start_time,
        shape=df.shape,
    )
    return df


def calculate_static_thresh(payload: StreamPayload, upper_limit: float):
    """Calculates anomaly scores using static thresholding."""
    x = payload.get_stream_array(original=True)
    static_clf = SigmoidThreshold(upper_limit=upper_limit)
    static_scores = static_clf.score_samples(x)
    return static_scores


class WindowScorer:
    """Class to calculate the final anomaly scores for the window.

    Args:
    ----
        metric_conf: MetricConf instance
    """

    __slots__ = ("static_wt", "static_limit", "model_wt", "postproc_clf")

    def __init__(self, metric_conf: MetricConf):
        self.static_wt = metric_conf.static_threshold_wt
        self.static_limit = metric_conf.static_threshold
        self.model_wt = 1.0 - self.static_wt

        postproc_factory = PostprocessFactory()
        self.postproc_clf = postproc_factory.get_instance(metric_conf.numalogic_conf.postprocess)

    def get_final_winscore(self, payload: StreamPayload) -> float:
        """Returns the final normalized window score.

        Performs soft voting ensembling if valid static threshold
        weight found in config.

        Args:
        ----
            payload: StreamPayload instance

        Returns
        -------
            Final score for the window
        """
        norm_winscore = self.get_winscore(payload)

        if not self.static_wt:
            return norm_winscore

        norm_static_winscore = self.get_static_winscore(payload)
        ensemble_score = (self.static_wt * norm_static_winscore) + (self.model_wt * norm_winscore)

        LOGGER.debug(
            "{uuid} - Model score: {m_score}, Static score: {s_score}, Static wt: {wt}",
            uuid=payload.uuid,
            m_score=norm_winscore,
            s_score=norm_static_winscore,
            wt=self.static_wt,
        )

        return ensemble_score

    def get_static_winscore(self, payload: StreamPayload) -> float:
        """Returns the normalized window score
        calculated using the static threshold estimator.

        Args:
        ----
            payload: StreamPayload instance

        Returns
        -------
            Score for the window
        """
        static_scores = calculate_static_thresh(payload, self.static_limit)
        static_winscore = np.mean(static_scores)
        return self.postproc_clf.transform(static_winscore)

    def get_winscore(self, payload: StreamPayload):
        """Returns the normalized window score.

        Args:
        ----
            payload: StreamPayload instance

        Returns
        -------
            Score for the window
        """
        scores = payload.get_stream_array()
        winscore = np.mean(scores)
        return self.postproc_clf.transform(winscore)

    def adjust_weights(self):
        """Adjust the soft voting weights depending on the streaming input."""
        raise NotImplementedError
