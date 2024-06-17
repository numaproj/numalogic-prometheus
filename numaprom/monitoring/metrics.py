import logging
from typing import Optional

from prometheus_client import Counter, Info, Summary, Gauge, Histogram

_LOGGER = logging.getLogger(__name__)


class BaseMetric:
    __slots__ = ("name", "label_pairs", "description")

    """
    Base class for metrics.

    Args:
        name: Name of the metric
        description: Description of the metric
        label_pairs: Labels in key, value pair
    """

    def __init__(
        self,
        name: str,
        description: str,
        label_pairs: dict,
    ) -> None:
        self.label_pairs = dict(label_pairs)  # converting DictConfig to dict type
        self.name = name
        self.description = description

    def add_info(self, labels: Optional[dict[str, str]], data: dict) -> None:
        pass

    def add_observation(self, labels: Optional[dict[str, str]], value: float) -> None:
        pass

    def increment_counter(self, labels: Optional[dict[str, str]], amount: int = 1) -> None:
        pass

    def set_gauge(self, labels: Optional[dict[str, str]], data: float) -> None:
        pass


class PromCounterMetric(BaseMetric):
    """Class is used to create a counter object and increment it."""

    __slots__ = "counter"

    def __init__(
        self,
        name: str,
        description: str,
        label_pairs: dict,
    ) -> None:
        """
        Args:
            name: Name of the metric
            description: Description of the metric
            label_pairs: dict of labels with their default value
        """
        super().__init__(name, description, label_pairs)
        self.counter = Counter(name, description, [*label_pairs.keys()])

    def increment_counter(self, labels: Optional[dict], amount: int = 1) -> None:
        if not labels:
            labels = {}
        _new_labels = self.label_pairs | labels
        self.counter.labels(**_new_labels).inc(amount=amount)


class PromInfoMetric(BaseMetric):
    """Class is used to create an info object and increment it."""

    __slots__ = "info"

    def __init__(
        self,
        name: str,
        description: str,
        label_pairs: dict,
    ) -> None:
        """
        Args:
            name:metric name
            description: Description of the metric
            label_pairs: dict of labels with their default value
        """
        super().__init__(name, description, label_pairs)
        self.info = Info(name, description, [*label_pairs.keys()])

    def add_info(
        self,
        labels: Optional[dict[str, str]],
        data: dict,
    ) -> None:
        if not labels:
            labels = {}
        _new_labels = self.label_pairs | labels
        self.info.labels(**_new_labels).info(data)


class PromSummaryMetric(BaseMetric):
    __slots__ = "summary"

    def __init__(
        self,
        name: str,
        description: str,
        label_pairs: dict,
    ) -> None:
        """
        Args:
            name: Name of the metric
            description: Description of the metric
            label_pairs: dict of labels with their default value
        """
        super().__init__(name, description, label_pairs)
        self.summary = Summary(name, description, [*label_pairs.keys()])

    def add_observation(self, labels: Optional[dict[str, str]], value: float) -> None:
        if not labels:
            labels = {}
        _new_labels = self.label_pairs | labels
        self.summary.labels(**_new_labels).observe(amount=value)


class PromGaugeMetric(BaseMetric):
    __slots__ = "gauge"

    def __init__(
        self,
        name: str,
        description: str,
        label_pairs: dict,
    ) -> None:
        """
        Args:
            name: Name of the metric
            description: Description of the metric
            label_pairs: dict of labels with their default value"""
        super().__init__(name, description, label_pairs)
        self.gauge = Gauge(name, description, [*label_pairs.keys()])

    def set_gauge(
        self,
        labels: Optional[dict[str, str]],
        data: float,
    ) -> None:
        if not labels:
            labels = {}
        _new_labels = self.label_pairs | labels
        self.gauge.labels(**_new_labels).set(data)


class PromHistogramMetric(BaseMetric):
    __slots__ = "histogram"

    def __init__(
        self,
        name: str,
        description: str,
        label_pairs: dict,
    ) -> None:
        """
        Args:
            name: Name of the metric
            description: Description of the metric
            label_pairs: dict of labels with their default value
        """
        super().__init__(name, description, label_pairs)
        self.histogram = Histogram(name, description, [*label_pairs.keys()])

    def add_observation(self, labels: Optional[dict[str, str]], value: float) -> None:
        if not labels:
            labels = {}
        _new_labels = self.label_pairs | labels
        self.histogram.labels(**_new_labels).observe(amount=value)
