from typing import List
from omegaconf import MISSING
from dataclasses import dataclass, field

from numalogic.config import NumalogicConf


@dataclass
class UnifiedConf:
    unified_metric_name: str
    unified_metrics: List[str]
    unified_strategy: str = "max"
    unified_weights: List[float] = field(default_factory=list)


@dataclass
class MetricConf:
    metric: str = "default"
    composite_keys: List[str] = field(default_factory=lambda: ["namespace", "name"])
    static_threshold: int = 3
    static_threshold_wt: float = 0.0
    scrape_interval: int = 30
    retrain_freq_hr: int = 8
    resume_training: bool = False
    numalogic_conf: NumalogicConf = MISSING


@dataclass
class AppConf:
    app: str = "default"
    namespace: str = "default"
    metric_configs: List[MetricConf] = field(default_factory=lambda: [MetricConf()])
    unified_configs: List[UnifiedConf] = field(default_factory=list)


@dataclass
class DataConf:
    configs: List[AppConf]


@dataclass
class RedisConf:
    host: str
    port: int
    auth: str
    expiry: int


@dataclass
class PrometheusConf:
    server: str


@dataclass
class RegistryConf:
    tracking_uri: str


@dataclass
class PipelineConf:
    redis_conf: RedisConf
    prometheus_conf: PrometheusConf
    registry_conf: RedisConf
