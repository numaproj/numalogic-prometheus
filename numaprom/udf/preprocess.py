import os
import time

import orjson
from numalogic.config import ModelInfo
from numalogic.registry import RedisRegistry, ArtifactData
from numalogic.tools.exceptions import RedisRegistryError
from pynumaflow.function import Datum, Messages, Message

from numaprom import get_logger
from numaprom.clients.sentinel import get_redis_client
from numaprom.entities import Status, StreamPayload, Header
from numaprom.watcher import ConfigManager

_LOGGER = get_logger(__name__)

AUTH = os.getenv("REDIS_AUTH")
REDIS_CONF = ConfigManager.get_redis_config()
REDIS_CLIENT = get_redis_client(
    REDIS_CONF.host,
    REDIS_CONF.port,
    password=AUTH,
    mastername=REDIS_CONF.master_name,
    recreate=False,
)


class Preprocess:
    """
    UDF class for data preprocessing.
    """

    __slots__ = ("model_registry",)

    def __init__(self, *_, **__):
        self.model_registry = RedisRegistry(client=REDIS_CLIENT)

    def __call__(self, keys: list[str], datum: Datum) -> Messages:
        return self.udf(keys, datum)

    @staticmethod
    def msg_forward(payload: StreamPayload) -> Messages:
        json_data = orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)
        if json_data:
            return Messages(Message(value=json_data))
        return Messages(Message.to_drop())

    @staticmethod
    def _run_preprocess(payload, artifact_data: ArtifactData) -> StreamPayload:
        # Perform preprocessing
        x_raw = payload.get_stream_array()
        preproc_clf = artifact_data.artifact
        x_scaled = preproc_clf.transform(x_raw)

        # Prepare payload for forwarding
        payload.set_win_arr(x_scaled)
        payload.set_status(Status.PRE_PROCESSED)
        return payload

    def _handle_registry_error(self, payload: StreamPayload, err: RedisRegistryError) -> Messages:
        """Handle registry error."""
        _LOGGER.exception(
            "%s - Error while fetching artifact, keys: %s, err: %r",
            payload.uuid,
            payload.composite_keys,
            err,
        )
        payload.set_header(Header.STATIC_INFERENCE)
        payload.set_status(Status.RUNTIME_ERROR)
        return self.msg_forward(payload)

    def _handle_not_found(self, payload: StreamPayload) -> Messages:
        """Handle artifact not found error."""
        _LOGGER.info(
            "%s - Preprocess artifact not found, forwarding for static thresholding. Keys: %s",
            payload.uuid,
            payload.composite_keys,
        )
        payload.set_header(Header.STATIC_INFERENCE)
        payload.set_status(Status.ARTIFACT_NOT_FOUND)
        return self.msg_forward(payload)

    @staticmethod
    def _get_conf(payload: StreamPayload) -> list[ModelInfo]:
        """
        Get preprocess config from config manager.
        """
        metric_config = ConfigManager.get_metric_config(payload.composite_keys)
        preprocess_cfgs = metric_config.numalogic_conf.preprocess
        return preprocess_cfgs

    def udf(self, _: list[str], datum: Datum) -> Messages:
        """
        UDF for preprocessing.
        """
        _start_time = time.perf_counter()
        _in_msg = datum.value.decode("utf-8")

        payload = StreamPayload(**orjson.loads(_in_msg))
        _LOGGER.info("%s - Received Payload: %r ", payload.uuid, payload)

        # Load config
        preprocess_cfgs = self._get_conf(payload)

        # Load preprocess artifact
        try:
            preproc_artifact = self.model_registry.load(
                skeys=[payload.composite_keys["namespace"], payload.composite_keys["name"]],
                dkeys=[_conf.name for _conf in preprocess_cfgs],
            )
        except RedisRegistryError as err:
            return self._handle_registry_error(payload, err)

        if not preproc_artifact:
            return self._handle_not_found(payload)

        payload = self._run_preprocess(payload, preproc_artifact)

        _LOGGER.info("%s - Sending Payload: %r ", payload.uuid, payload)
        _LOGGER.debug(
            "%s - Time taken in preprocess: %.4f sec",
            payload.uuid,
            time.perf_counter() - _start_time,
        )
        return self.msg_forward(payload)
