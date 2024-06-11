from dataclasses import dataclass


@dataclass
class PromMetric:
    name: str
    description: str
    static_labels_pair: dict[str, str] | None = None
    labels_pair: dict[str, str] | None = None


@dataclass
class PromMetricList:
    type: str
    metrics: list[PromMetric]
