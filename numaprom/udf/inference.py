import os
import time
from typing import Final

from numalogic.config import NumalogicConf
from numalogic.models.autoencoder import AutoencoderTrainer
from numalogic.registry import ArtifactData, RedisRegistry, LocalLRUCache
from numalogic.tools.data import StreamingDataset
from numalogic.tools.exceptions import RedisRegistryError
from orjson import orjson
from pynumaflow.mapper import Datum
from torch.utils.data import DataLoader

from numaprom import LOGGER
from numaprom.clients.sentinel import get_redis_client_from_conf
from numaprom.entities import PayloadFactory
from numaprom.entities import Status, StreamPayload, Header
from numaprom.metrics import increase_redis_conn_error, inc_inference_count
from numaprom.tools import msg_forward
from numaprom.watcher import ConfigManager


_VERTEX: Final[str] = "inference"
REDIS_CLIENT = get_redis_client_from_conf(master_node=False)
LOCAL_CACHE_TTL = int(os.getenv("LOCAL_CACHE_TTL", 3600))  # default ttl set to 1 hour


def _run_inference(
    payload: StreamPayload, artifact_data: ArtifactData, numalogic_conf: NumalogicConf
) -> StreamPayload:
    model = artifact_data.artifact
    stream_data = payload.get_stream_array()
    stream_loader = DataLoader(StreamingDataset(stream_data, numalogic_conf.model.conf["seq_len"]))

    trainer = AutoencoderTrainer()
    try:
        recon_err = trainer.predict(model, dataloaders=stream_loader)
    except Exception as err:
        LOGGER.exception(
            "{uuid} - Runtime error while performing inference: {err}", uuid=payload.uuid, err=err
        )
        raise RuntimeError("Failed to infer") from err

    LOGGER.info("{uuid} - Successfully inferred", uuid=payload.uuid)

    payload.set_win_arr(recon_err.numpy())
    payload.set_status(Status.INFERRED)
    payload.set_metadata("version", artifact_data.extras.get("version"))
    return payload


@msg_forward
def inference(_: list[str], datum: Datum) -> bytes:
    _start_time = time.perf_counter()

    _in_msg = datum.value.decode("utf-8")

    # Construct payload object
    payload = PayloadFactory.from_json(_in_msg)

    # Check if payload needs static inference
    if payload.header == Header.STATIC_INFERENCE:
        LOGGER.debug(
            "{uuid} - Models not found in the previous steps, forwarding for "
            "static thresholding. Keys: {keys}",
            uuid=payload.uuid,
            keys=payload.composite_keys,
        )
        return orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)

    # Load config
    metric_config = ConfigManager.get_metric_config(payload.composite_keys)
    numalogic_conf = metric_config.numalogic_conf

    # Load inference model
    local_cache = LocalLRUCache(ttl=LOCAL_CACHE_TTL)
    model_registry = RedisRegistry(client=REDIS_CLIENT, cache_registry=local_cache)
    try:
        artifact_data = model_registry.load(
            skeys=[payload.composite_keys["namespace"], payload.composite_keys["name"]],
            dkeys=[numalogic_conf.model.name],
        )
    except RedisRegistryError as err:
        LOGGER.exception(
            "{uuid} - Error while fetching inference artifact, keys: {keys}, err: {err}",
            uuid=payload.uuid,
            keys=payload.composite_keys,
            err=err,
        )
        payload.set_header(Header.STATIC_INFERENCE)
        payload.set_status(Status.RUNTIME_ERROR)
        increase_redis_conn_error(_VERTEX)
        return orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)

    if not artifact_data:
        LOGGER.info(
            "{uuid} - Inference artifact not found, "
            "forwarding for static thresholding. Keys: {keys}",
            uuid=payload.uuid,
            keys=payload.composite_keys,
        )
        payload.set_header(Header.STATIC_INFERENCE)
        payload.set_status(Status.ARTIFACT_NOT_FOUND)
        return orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)

    LOGGER.info(
        "{uuid} - Loaded artifact data from {source} ",
        uuid=payload.uuid,
        source=artifact_data.extras.get("source"),
    )

    # Check if current model is stale and source is 'registry'
    if (
        RedisRegistry.is_artifact_stale(artifact_data, int(metric_config.retrain_freq_hr))
        and artifact_data.extras.get("source") == "registry"
    ):
        payload.set_header(Header.MODEL_STALE)

    # Generate predictions
    try:
        payload = _run_inference(payload, artifact_data, numalogic_conf)
    except RuntimeError:
        LOGGER.info(
            "{uuid} - Failed to infer, forwarding for static thresholding. Keys: {keys}",
            uuid=payload.uuid,
            keys=payload.composite_keys,
        )
        payload.set_header(Header.STATIC_INFERENCE)
        payload.set_status(Status.RUNTIME_ERROR)
        return orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)

    inc_inference_count(
        model=payload.get_metadata("version"),
        namespace=payload.composite_keys.get("namespace"),
        app=payload.composite_keys.get("app"),
        metric=payload.composite_keys.get("name"),
        status=payload.header,
    )
    LOGGER.info("{uuid} - Sending Payload: {payload} ", uuid=payload.uuid, payload=payload)
    LOGGER.debug(
        "{uuid} - Time taken in inference: {time} sec",
        uuid=payload.uuid,
        time=time.perf_counter() - _start_time,
    )
    return orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)
