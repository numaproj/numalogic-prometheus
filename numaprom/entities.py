from copy import copy
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional, Any, Union

import numpy as np
import numpy.typing as npt
import orjson

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

    def set_metadata(self, key: str, value):
        self.metadata[key] = value

    def __repr__(self):
        return "{uuid: %s, win_arr: %s, win_ts_arr: %s, composite_keys: %s, metadata: %s}" % (
            self.uuid,
            list(self.win_arr),
            self.win_ts_arr,
            self.composite_keys,
            self.metadata,
        )


@dataclass
class PrometheusPayload:
    timestamp_ms: int
    name: str
    namespace: str
    subsystem: Optional[str]
    type: str
    value: float
    labels: Dict[str, str]

    def as_json(self) -> bytes:
        return orjson.dumps(
            {
                "TimestampMs": self.timestamp_ms,
                "Name": self.name,
                "Namespace": self.namespace,
                "Subsystem": self.subsystem,
                "Type": self.type,
                "Value": self.value,
                "Labels": self.labels,
            }
        )

    @classmethod
    def from_json(cls, json_obj) -> "PrometheusPayload":
        obj = orjson.loads(json_obj)
        return cls(
            timestamp_ms=obj["TimestampMs"],
            name=obj["Name"],
            namespace=obj["Namespace"],
            subsystem=obj["Subsystem"],
            type=obj["Type"],
            value=obj["Value"],
            labels=obj["Labels"],
        )

    def __repr__(self):
        return (
            "{timestamp_ms: %s, name: %s, namespace: %s, "
            "subsystem: %s, type: %s, value: %s, labels: %s}"
            % (
                self.timestamp_ms,
                self.name,
                self.namespace,
                self.subsystem,
                self.type,
                self.value,
                self.labels,
            )
        )
