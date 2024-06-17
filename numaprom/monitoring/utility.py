# Copyright 2022 The Numaproj Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
from typing import Union, Optional

from numaprom.monitoring.metrics import (
    PromCounterMetric,
    PromInfoMetric,
    PromSummaryMetric,
    PromGaugeMetric,
    PromHistogramMetric,
)

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


def get_metric(
    metric_type: str,
    name: str,
    description: str,
    label_pairs: Optional[dict[str, str]],
) -> Union[
    PromCounterMetric, PromInfoMetric, PromSummaryMetric, PromGaugeMetric, PromHistogramMetric
]:
    """
    Returns a Prometheus metric object based on the metric type.
    Args:
        metric_type: metric type
        name: metric name
        description: metric description
        label_pairs: label pairs
    Returns: _BaseMetric covariant type
    """
    if metric_type == "Counter":
        return PromCounterMetric(name, description, label_pairs)
    if metric_type == "Info":
        return PromInfoMetric(name, description, label_pairs)
    if metric_type == "Summary":
        return PromSummaryMetric(name, description, label_pairs)
    if metric_type == "Gauge":
        return PromGaugeMetric(name, description, label_pairs)
    if metric_type == "Histogram":
        return PromHistogramMetric(name, description, label_pairs)
    raise ValueError(f"Unknown metric type: {metric_type}")
