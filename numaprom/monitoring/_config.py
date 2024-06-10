from dataclasses import dataclass
from typing import Optional


@dataclass
class NumalogicMetric:
    name: str
    description: str
    static_labels_pair: Optional[dict[str, str]] = None
    labels_pair: Optional[dict[str, str]] = None


@dataclass
class NumalogicMetricTypeConfig:
    type: str
    metrics: list[NumalogicMetric]
