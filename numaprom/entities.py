from copy import copy
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional, Any, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from dataclasses_json import dataclass_json, LetterCase

Vector = List[float]
Matrix = Union[Vector, List[Vector], npt.NDArray]


class Status(str, Enum):
    RAW = "raw"
    EXTRACTED = "extracted"
    PRE_PROCESSED = "pre_processed"
    INFERRED = "inferred"
    POST_PROCESSED = "post_processed"


@dataclass(frozen=True)
class Metric:
    timestamp: str
    value: float


@dataclass
class StreamPayload:
    uuid: str
    win_arr: Matrix
    win_ts_arr: List[str]
    composite_keys: Dict[str, str]
    status: Status = Status.RAW
    metadata: Dict[str, Any] = None

    @property
    def start_ts(self):
        return self.win_ts_arr[0]

    @property
    def end_ts(self):
        return self.win_ts_arr[-1]

    def get_streamarray(self):
        return np.asarray(self.win_arr)

    def get_metadata(self, key: str):
        return copy(self.metadata[key])

    def set_win_arr(self, arr):
        self.win_arr = arr

    def set_status(self, status: Status):
        self.status = status


@dataclass_json
@dataclass
class Payload:
    uuid: str
    metric_name: str
    key_map: Dict
    src_labels: Dict[str, str]
    processedMetrics: List[Metric]
    startTS: str
    endTS: str
    status: Status = Status.RAW
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
