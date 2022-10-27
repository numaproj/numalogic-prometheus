import numpy as np
import pandas as pd
from enum import Enum
from typing import List, Dict, Optional
from dataclasses import dataclass
from dataclasses_json import dataclass_json, LetterCase


class MetricType(Enum):
    LATENCY = "latency"
    CPU = "cpu"
    MEMORY = "memory"
    ERROR_RATE = "error_rate"
    ERROR_COUNT = "error_count"
    HASH_ERROR_RATE = "hash_error_rate"
    HASH_LATENCY = "hash_latency"


class Status(str, Enum):
    RAW = "raw"
    EXTRACTED = "extracted"
    PRE_PROCESSED = "pre_processed"
    INFERRED = "inferred"
    POST_PROCESSED = "post_processed"


@dataclass_json
@dataclass
class Metric:
    timestamp: str
    value: float = 0


@dataclass_json
@dataclass
class Payload:
    uuid: str
    namespace: str
    metric: str
    src_labels: Dict[str, str]
    inputMetrics: List[Metric]
    processedMetrics: List[Metric]
    startTS: str
    endTS: str
    status: Status = Status.RAW
    hash_id: str = None
    win_score: Optional[List[float]] = None
    steps: Optional[Dict[str, str]] = None
    std: Optional[float] = None
    mean: Optional[float] = None
    threshold: Optional[float] = None
    model_version: Optional[int] = None
    anomaly: Optional[float] = None

    def get_processed_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame([metric.__dict__ for metric in self.processedMetrics])
        df.set_index("timestamp", inplace=True)
        return df

    def get_input_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame([metric.__dict__ for metric in self.inputMetrics])
        df.set_index("timestamp", inplace=True)
        return df

    def get_processed_array(self) -> np.array:
        arr = np.array([[metric.value] for metric in self.processedMetrics])
        return arr

    def __eq__(self, other):
        return isinstance(other, Payload) and self.uuid == other.uuid

    def __hash__(self):
        return hash(self.uuid)


@dataclass_json(letter_case=LetterCase.PASCAL)
@dataclass
class PrometheusPayload:
    timestamp_ms: int
    name: str
    namespace: str
    subsystem: Optional[str]
    type: str
    value: float
    labels: Dict[str, str]
