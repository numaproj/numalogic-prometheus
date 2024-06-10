import logging
from typing import Optional
from collections.abc import Sequence

from prometheus_client import Counter, Info, Summary, Gauge, Histogram

_LOGGER = logging.getLogger(__name__)


class _BaseMetric:
    __slots__ = ("name", "description", "labels")

    """
    Base class for metrics.

    Args:
        name: Name of the metric
        description: Description of the metric
        labels: List of labels
    """

    def __init__(self, name: str, description: str, labels: Optional[Sequence[str]]) -> None:
        self.name = name
        self.description = description
        self.labels = labels

    def add_info(self, labels: Optional[dict], data: dict) -> None:
        pass

    def add_observation(self, labels: Optional[dict], value: float) -> None:
        pass

    def increment_counter(self, labels: Optional[dict], amount: int = 1) -> None:
        pass

    def set_gauge(self, labels: dict, data: float) -> None:
        pass


class PromCounterMetric(_BaseMetric):
    """Class is used to create a counter object and increment it."""

    __slots__ = ("counter", "static_label_pairs")

    def __init__(
        self,
        name: str,
        description: str,
        label_pairs: dict[str, str],
        static_label_pairs: dict[str, str],
    ) -> None:
        super().__init__(name, description, [*label_pairs.keys(), *static_label_pairs.keys()])
        self.counter = Counter(name, description, [*label_pairs.keys(), *static_label_pairs.keys()])
        self.static_label_pairs = dict(static_label_pairs)  # converting DictConfig to dict type

    def increment_counter(self, labels: Optional[dict], amount: int = 1) -> None:
        _new_labels = self.static_label_pairs | labels
        self.counter.labels(**_new_labels).inc(amount=amount)


class PromInfoMetric(_BaseMetric):
    """Class is used to create an info object and increment it."""

    __slots__ = ("info", "static_label_pairs")

    def __init__(
        self,
        name: str,
        description: str,
        label_pairs: dict[str, str],
        static_label_pairs: dict[str, str],
    ) -> None:
        super().__init__(name, description, [*label_pairs.keys(), *static_label_pairs.keys()])
        self.info = Info(name, description, [*label_pairs.keys(), *static_label_pairs.keys()])
        self.static_label_pairs = dict(static_label_pairs)  # converting DictConfig to dict type

    def add_info(
        self,
        labels: Optional[dict],
        data: dict,
    ) -> None:
        _new_labels = self.static_label_pairs | labels
        self.info.labels(**_new_labels).info(data)


class PromSummaryMetric(_BaseMetric):
    __slots__ = ("summary", "static_label_pairs")

    def __init__(
        self,
        name: str,
        description: str,
        label_pairs: dict[str, str],
        static_label_pairs: dict[str, str],
    ) -> None:
        super().__init__(name, description, [*label_pairs.keys(), *static_label_pairs.keys()])
        self.summary = Summary(name, description, [*label_pairs.keys(), *static_label_pairs.keys()])
        self.static_label_pairs = dict(static_label_pairs)  # converting DictConfig to dict type

    def add_observation(self, labels: Optional[dict], value: float) -> None:
        _new_labels = self.static_label_pairs | labels
        self.summary.labels(**_new_labels).observe(amount=value)


class PromGaugeMetric(_BaseMetric):
    __slots__ = ("info", "static_label_pairs")

    def __init__(
        self,
        name: str,
        description: str,
        label_pairs: dict[str, str],
        static_label_pairs: dict[str, str],
    ) -> None:
        super().__init__(name, description, [*label_pairs.keys(), *static_label_pairs.keys()])
        self.info = Gauge(name, description, [*label_pairs.keys(), *static_label_pairs.keys()])
        self.static_label_pairs = dict(static_label_pairs)  # converting DictConfig to dict type

    def set_gauge(
        self,
        labels: dict,
        data: float,
    ) -> None:
        _new_labels = self.static_label_pairs | labels
        self.info.labels(**_new_labels).set(data)


class PromHistogramMetric(_BaseMetric):
    __slots__ = ("histogram", "static_label_pairs")

    def __init__(
        self,
        name: str,
        description: str,
        label_pairs: dict[str, str],
        static_label_pairs: dict[str, str],
    ) -> None:
        super().__init__(name, description, [*label_pairs.keys(), *static_label_pairs.keys()])
        self.histogram = Histogram(
            name, description, [*label_pairs.keys(), *static_label_pairs.keys()]
        )
        self.static_label_pairs = dict(static_label_pairs)  # converting DictConfig to dict type

    def add_observation(self, labels: Optional[dict], value: float) -> None:
        _new_labels = self.static_label_pairs | labels
        self.histogram.labels(**_new_labels).observe(amount=value)
