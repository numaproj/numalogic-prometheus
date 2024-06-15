import logging

from prometheus_client import Counter, Info, Summary, Gauge, Histogram

_LOGGER = logging.getLogger(__name__)


class BaseMetric:
    __slots__ = ("name", "description", "static_label_pairs", "label_pairs")

    """
    Base class for metrics.

    Args:
        name: Name of the metric
        description: Description of the metric
        labels: List of labels
    """

    def __init__(
        self,
        name: str,
        description: str,
        static_label_pairs: dict[str, str] | None,
        label_pairs: dict[str, str] | None,
    ) -> None:
        self.label_pairs = label_pairs
        self.static_label_pairs = dict(static_label_pairs)  # converting DictConfig to dict type
        self.name = name
        self.description = description

    def add_info(self, labels: dict | None, data: dict) -> None:
        pass

    def add_observation(self, labels: dict | None, value: float) -> None:
        pass

    def increment_counter(self, labels: dict | None, amount: int = 1) -> None:
        pass

    def set_gauge(self, labels: dict, data: float) -> None:
        pass


class PromCounterMetric(BaseMetric):
    """Class is used to create a counter object and increment it."""

    __slots__ = ("counter", "static_label_pairs")

    def __init__(
        self,
        name: str,
        description: str,
        label_pairs: dict[str, str],
        static_label_pairs: dict[str, str],
    ) -> None:
        """
        Args:
            name: Name of the metric
            description: Description of the metric
            label_pairs: dict of labels with their default value
            static_label_pairs: dict of static labels with the default value
        """
        super().__init__(name, description, static_label_pairs, label_pairs)
        self.counter = Counter(name, description, [*label_pairs.keys(), *static_label_pairs.keys()])

    def increment_counter(self, labels: dict | None, amount: int = 1) -> None:
        _new_labels = self.static_label_pairs | labels
        self.counter.labels(**_new_labels).inc(amount=amount)


class PromInfoMetric(BaseMetric):
    """Class is used to create an info object and increment it."""

    __slots__ = ("info", "static_label_pairs")

    def __init__(
        self,
        name: str,
        description: str,
        label_pairs: dict[str, str],
        static_label_pairs: dict[str, str],
    ) -> None:
        """
        Args:
            name:metric name
            description: Description of the metric
            label_pairs: dict of labels with their default value
            static_label_pairs: dict of static labels with the default value
        """
        super().__init__(name, description, static_label_pairs, label_pairs)
        self.info = Info(name, description, [*label_pairs.keys(), *static_label_pairs.keys()])

    def add_info(
        self,
        labels: dict | None,
        data: dict,
    ) -> None:
        _new_labels = self.static_label_pairs | labels
        self.info.labels(**_new_labels).info(data)


class PromSummaryMetric(BaseMetric):
    __slots__ = ("summary", "static_label_pairs")

    def __init__(
        self,
        name: str,
        description: str,
        label_pairs: dict[str, str],
        static_label_pairs: dict[str, str],
    ) -> None:
        """
        Args:
            name: Name of the metric
            description: Description of the metric
            label_pairs: dict of labels with their default value
            static_label_pairs: dict of static labels with the default value
        """
        super().__init__(name, description, static_label_pairs, label_pairs)
        self.summary = Summary(name, description, [*label_pairs.keys(), *static_label_pairs.keys()])

    def add_observation(self, labels: dict | None, value: float) -> None:
        _new_labels = self.static_label_pairs | labels
        self.summary.labels(**_new_labels).observe(amount=value)


class PromGaugeMetric(BaseMetric):
    __slots__ = ("info", "static_label_pairs")

    def __init__(
        self,
        name: str,
        description: str,
        label_pairs: dict[str, str],
        static_label_pairs: dict[str, str],
    ) -> None:
        """
        Args:
            name: Name of the metric
            description: Description of the metric
            label_pairs: dict of labels with their default value
            static_label_pairs: dict of static labels with the default value
        """
        super().__init__(name, description, static_label_pairs, label_pairs)
        self.info = Gauge(name, description, [*label_pairs.keys(), *static_label_pairs.keys()])

    def set_gauge(
        self,
        labels: dict,
        data: float,
    ) -> None:
        _new_labels = self.static_label_pairs | labels
        self.info.labels(**_new_labels).set(data)


class PromHistogramMetric(BaseMetric):
    __slots__ = ("histogram", "static_label_pairs")

    def __init__(
        self,
        name: str,
        description: str,
        label_pairs: dict[str, str],
        static_label_pairs: dict[str, str],
    ) -> None:
        """
        Args:
            name: Name of the metric
            description: Description of the metric
            label_pairs: dict of labels with their default value
            static_label_pairs: dict of static labels with the default value
        """
        super().__init__(name, description, static_label_pairs, label_pairs)
        self.histogram = Histogram(
            name, description, [*label_pairs.keys(), *static_label_pairs.keys()]
        )

    def add_observation(self, labels: dict | None, value: float) -> None:
        _new_labels = self.static_label_pairs | labels
        self.histogram.labels(**_new_labels).observe(amount=value)
