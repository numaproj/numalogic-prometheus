from copy import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar
from collections import OrderedDict

import numpy as np
import numpy.typing as npt
import orjson
from typing_extensions import Self

Vector = list[float]
Matrix = Vector | list[Vector] | npt.NDArray[float]
np.set_printoptions(precision=3, suppress=True)


class Status(str, Enum):
    RAW = "raw"
    EXTRACTED = "extracted"
    PRE_PROCESSED = "pre_processed"
    INFERRED = "inferred"
    THRESHOLD = "threshold_complete"
    POST_PROCESSED = "post_processed"
    ARTIFACT_NOT_FOUND = "artifact_not_found"
    ARTIFACT_STALE = "artifact_is_stale"
    RUNTIME_ERROR = "runtime_error"


class Header(str, Enum):
    STATIC_INFERENCE = "static_threshold"
    MODEL_INFERENCE = "model_inference"
    TRAIN_REQUEST = "request_training"
    MODEL_STALE = "model_stale"


@dataclass(slots=True)
class _BasePayload:
    uuid: str
    composite_keys: OrderedDict[str, str]


PayloadType = TypeVar("PayloadType", bound=_BasePayload)


@dataclass(slots=True)
class TrainerPayload(_BasePayload):
    header: Header = Header.TRAIN_REQUEST


@dataclass(slots=True, repr=False)
class StreamPayload(_BasePayload):
    win_raw_arr: Matrix
    win_arr: Matrix
    win_ts_arr: list[str]
    status: Status = Status.RAW
    metadata: dict[str, Any] = field(default_factory=dict)
    header: Header = Header.MODEL_INFERENCE

    @property
    def start_ts(self) -> str:
        return self.win_ts_arr[0]

    @property
    def end_ts(self) -> str:
        return self.win_ts_arr[-1]

    def get_stream_array(self, original=False) -> npt.NDArray[float]:
        if original:
            return np.array(self.win_raw_arr)
        return np.array(self.win_arr)

    def get_metadata(self, key: str) -> dict[str, Any]:
        return copy(self.metadata[key])

    def set_win_arr(self, arr: Matrix) -> None:
        self.win_arr = arr

    def set_status(self, status: Status) -> None:
        self.status = status

    def set_header(self, header: Header) -> None:
        self.header = header

    def set_metadata(self, key: str, value) -> None:
        self.metadata[key] = value

    def __repr__(self) -> str:
        return (
            "header: {}, win_raw_arr: {}, win_arr: {}, win_ts_arr: {}, composite_keys:"
            " {}, metadata: {}}}".format(
                self.header,
                list(self.win_raw_arr),
                list(self.win_arr),
                self.win_ts_arr,
                self.composite_keys,
                self.metadata,
            )
        )


class PayloadFactory:
    __HEADER_MAP = {
        Header.MODEL_INFERENCE: StreamPayload,
        Header.TRAIN_REQUEST: TrainerPayload,
        Header.STATIC_INFERENCE: StreamPayload,
        Header.MODEL_STALE: StreamPayload,
    }

    @classmethod
    def from_json(cls, json_data: bytes | str) -> PayloadType:
        data = orjson.loads(json_data)
        header = data.get("header")
        if not header:
            raise RuntimeError(f"Header not present in json: {json_data}")
        payload_cls = cls.__HEADER_MAP[header]
        return payload_cls(**data)


@dataclass(slots=True, repr=False)
class PrometheusPayload:
    timestamp_ms: int
    name: str
    namespace: str
    subsystem: str | None
    type: str
    value: float
    labels: dict[str, str]

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
            },
            option=orjson.OPT_SERIALIZE_NUMPY,
        )

    @classmethod
    def from_json(cls, json_obj: bytes | str) -> Self:
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

    def __repr__(self) -> str:
        return (
            "{{timestamp_ms: {}, name: {}, namespace: {}, "
            "subsystem: {}, type: {}, value: {}, labels: {}}}".format(
                self.timestamp_ms,
                self.name,
                self.namespace,
                self.subsystem,
                self.type,
                self.value,
                self.labels,
            )
        )
