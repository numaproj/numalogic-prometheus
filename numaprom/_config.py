from omegaconf import MISSING
from dataclasses import dataclass, field

from numalogic.config import NumalogicConf


@dataclass
class UnifiedConf:
    unified_metric_name: str
    unified_metrics: list[str]
    unified_strategy: str = "max"
    unified_weights: list[float] = field(default_factory=list)


@dataclass
class MetricConf:
    metric: str = "default"
    composite_keys: list[str] = field(default_factory=lambda: ["namespace", "name"])
    static_threshold: int = 3
    static_threshold_wt: float = 0.0
    train_hours: int = 24 * 8  # 8 days worth of data
    min_train_size: int = 2000
    retrain_freq_hr: int = 24
    resume_training: bool = False
    scrape_interval: int = 30
    numalogic_conf: NumalogicConf = MISSING


@dataclass
class AppConf:
    app: str = "default"
    namespace: str = "default"
    metric_configs: list[MetricConf] = field(default_factory=lambda: [MetricConf()])
    unified_configs: list[UnifiedConf] = field(default_factory=list)


@dataclass
class DataConf:
    configs: list[AppConf]


@dataclass
class RedisConf:
    host: str
    port: int
    expiry: int = 300
    master_name: str = "mymaster"


@dataclass
class PrometheusConf:
    server: str
    pushgateway: str


@dataclass
class RegistryConf:
    tracking_uri: str


@dataclass
class PipelineConf:
    redis_conf: RedisConf
    prometheus_conf: PrometheusConf
    registry_conf: RegistryConf
