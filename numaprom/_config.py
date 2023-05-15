from typing import List
from omegaconf import MISSING
from dataclasses import dataclass
from pydantic import BaseModel, Field
from numalogic.config import NumalogicConf


@dataclass
class UnifiedConf(BaseModel):
    unified_metric_name: str
    unified_metrics: List[str]
    unified_strategy: str = "max"
    unified_weights: List[float] = Field(default_factory=list)


@dataclass
class MetricConf(BaseModel):
    metric: str = "default"
    composite_keys: List[str] = Field(default_factory=lambda: ["namespace", "name"])
    static_threshold: int = 3
    static_threshold_wt: float = 0.0
    train_hours: int = 36
    min_train_size: int = 2000
    retrain_freq_hr: int = 8
    resume_training: bool = False
    scrape_interval: int = 30
    numalogic_conf: NumalogicConf = MISSING


@dataclass
class AppConf(BaseModel):
    app: str = "default"
    namespace: str = "default"
    metric_configs: List[MetricConf] = Field(default_factory=lambda: [MetricConf()])
    unified_configs: List[UnifiedConf] = Field(default_factory=list)


@dataclass
class DataConf(BaseModel):
    configs: List[AppConf]


@dataclass
class RedisConf(BaseModel):
    host: str
    port: int
    expiry: int = 300
    master_name: str = "mymaster"


@dataclass
class PrometheusConf(BaseModel):
    server: str
    pushgateway: str


@dataclass
class RegistryConf(BaseModel):
    tracking_uri: str


@dataclass
class PipelineConf(BaseModel):
    redis_conf: RedisConf
    prometheus_conf: PrometheusConf
    registry_conf: RegistryConf
