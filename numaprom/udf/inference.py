import os
import time

from numalogic.config import NumalogicConf
from numalogic.models.autoencoder import AutoencoderTrainer
from numalogic.registry import ArtifactData, RedisRegistry
from numalogic.tools.data import StreamingDataset
from numalogic.tools.exceptions import RedisRegistryError
from orjson import orjson
from pynumaflow.function import Datum
from torch.utils.data import DataLoader

from numaprom import get_logger
from numaprom.clients.sentinel import get_redis_client
from numaprom.entities import PayloadFactory
from numaprom.entities import Status, StreamPayload, Header
from numaprom.tools import msg_forward
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
        _LOGGER.exception("%s - Runtime error while performing inference: %r", payload.uuid, err)
        raise RuntimeError("Failed to infer") from err

    _LOGGER.info("%s - Successfully inferred", payload.uuid)

    payload.set_win_arr(recon_err.numpy())
    payload.set_status(Status.INFERRED)
    payload.set_metadata("version", artifact_data.extras.get("version"))
    return payload


@msg_forward
def inference(_: str, datum: Datum) -> bytes:
    _start_time = time.perf_counter()

    _in_msg = datum.value.decode("utf-8")

    # Construct payload object
    payload = PayloadFactory.from_json(_in_msg)

    # Check if payload needs static inference
    if payload.header == Header.STATIC_INFERENCE:
        _LOGGER.debug(
            "%s - Models not found in the previous steps, forwarding for static thresholding. Keys: %s",
            payload.uuid,
            payload.composite_keys,
        )
        return orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)

    # Load config
    metric_config = ConfigManager.get_metric_config(payload.composite_keys)
    numalogic_conf = metric_config.numalogic_conf

    # Load inference model
    model_registry = RedisRegistry(client=REDIS_CLIENT)
    try:
        artifact_data = model_registry.load(
            skeys=[payload.composite_keys["namespace"], payload.composite_keys["name"]],
            dkeys=[numalogic_conf.model.name],
        )
    except RedisRegistryError as err:
        _LOGGER.exception(
            "%s - Error while fetching inference artifact, keys: %s, err: %r",
            payload.uuid,
            payload.composite_keys,
            err,
        )
        payload.set_header(Header.STATIC_INFERENCE)
        payload.set_status(Status.RUNTIME_ERROR)
        return orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)

    if not artifact_data:
        _LOGGER.info(
            "%s - Inference artifact not found, forwarding for static thresholding. Keys: %s",
            payload.uuid,
            payload.composite_keys,
        )
        payload.set_header(Header.STATIC_INFERENCE)
        payload.set_status(Status.ARTIFACT_NOT_FOUND)
        return orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)

    # Check if current model is stale
    if RedisRegistry.is_artifact_stale(artifact_data, int(metric_config.retrain_freq_hr)):
        payload.set_header(Header.MODEL_STALE)

    # Generate predictions
    try:
        payload = _run_inference(payload, artifact_data, numalogic_conf)
    except RuntimeError:
        _LOGGER.info(
            "%s - Failed to infer, forwarding for static thresholding. Keys: %s",
            payload.uuid,
            payload.composite_keys,
        )
        payload.set_header(Header.STATIC_INFERENCE)
        payload.set_status(Status.RUNTIME_ERROR)
        return orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)

    _LOGGER.info("%s - Sending Payload: %s ", payload.uuid, payload)
    _LOGGER.debug(
        "%s - Time taken in inference: %.4f sec", payload.uuid, time.perf_counter() - _start_time
    )
    return orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)
