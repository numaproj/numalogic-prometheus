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
from typing import Optional

from omegaconf import OmegaConf

from numaprom.monitoring.metrics import PromCounterMetric, PromInfoMetric, PromSummaryMetric, PromGaugeMetric, \
    PromHistogramMetric

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


def _get_metric(
        metric_type: str,
        name: str,
        description: str,
        label_pairs: Optional[dict[str, str]],
        static_label_pairs: Optional[dict[str, str]],
) -> PromCounterMetric | PromInfoMetric | PromSummaryMetric | PromGaugeMetric | PromHistogramMetric:
    if metric_type == "Counter":
        return PromCounterMetric(name, description, label_pairs, static_label_pairs)
    if metric_type == "Info":
        return PromInfoMetric(name, description, label_pairs, static_label_pairs)
    if metric_type == "Summary":
        return PromSummaryMetric(name, description, label_pairs, static_label_pairs)
    if metric_type == "Gauge":
        return PromGaugeMetric(name, description, label_pairs, static_label_pairs)
    if metric_type == "Histogram":
        return PromHistogramMetric(name, description, label_pairs, static_label_pairs)
    raise ValueError(f"Unknown metric type: {metric_type}")


def create_metrics_from_config_file(config_file_path: str) -> dict[str, _BaseMetric]:
    config = OmegaConf.load(config_file_path)
    metrics = {}
    for metric_config in config.get("numalogic_metrics", []):
        metric_type = metric_config["type"]
        for metric in metric_config["metrics"]:
            name = metric["name"]
            description = metric.get("description", "")
            label_pairs = metric.get("label_pairs", {})
            static_label_pairs = metric.get("static_label_pairs", {})
            metrics[name] = _get_metric(
                metric_type, name, description, label_pairs, static_label_pairs
            )
    return metrics
