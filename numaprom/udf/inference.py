import time
from datetime import datetime, timedelta

from numalogic.config import NumalogicConf
from numalogic.models.autoencoder import AutoencoderTrainer
from numalogic.registry import ArtifactData
from numalogic.tools.data import StreamingDataset
from orjson import orjson
from pynumaflow.function import Datum
from torch.utils.data import DataLoader

from numaprom import get_logger, MetricConf
from numaprom.entities import PayloadFactory
from numaprom.entities import Status, StreamPayload, Header
from numaprom.tools import (
    load_model,
    get_metric_config,
    msg_forward,
)

_LOGGER = get_logger(__name__)


def _run_inference(
    payload: StreamPayload, artifact_data: ArtifactData, numalogic_conf: NumalogicConf
) -> StreamPayload:
    model = artifact_data.artifact
    stream_data = payload.get_stream_array()
    stream_loader = DataLoader(StreamingDataset(stream_data, numalogic_conf.model.conf["seq_len"]))

    trainer = AutoencoderTrainer()
    recon_err = trainer.predict(model, dataloaders=stream_loader)

    _LOGGER.info("%s - Successfully inferred", payload.uuid)

    payload.set_win_arr(recon_err.numpy())
    payload.set_status(Status.INFERRED)
    payload.set_metadata("version", artifact_data.extras.get("version"))
    return payload


def _is_model_stale(
    payload: StreamPayload, artifact_data: ArtifactData, metric_config: MetricConf
) -> bool:
    date_updated = artifact_data.extras["last_updated_timestamp"] / 1000
    stale_date = (datetime.now() - timedelta(hours=int(metric_config.retrain_freq_hr))).timestamp()
    if date_updated < stale_date:
        _LOGGER.info(
            "%s - Model found is stale for %s",
            payload.uuid,
            payload.composite_keys,
        )
        return True
    return False


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
    metric_config = get_metric_config(
        metric=payload.composite_keys["name"], namespace=payload.composite_keys["namespace"]
    )
    numalogic_conf = metric_config.numalogic_conf

    # Load inference model
    artifact_data = load_model(
        skeys=[payload.composite_keys["namespace"], payload.composite_keys["name"]],
        dkeys=[numalogic_conf.model.name],
    )
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
    if _is_model_stale(payload, artifact_data, metric_config):
        payload.set_header(Header.MODEL_STALE)

    # Generate predictions
    payload = _run_inference(payload, artifact_data, numalogic_conf)

    _LOGGER.info("%s - Sending Payload: %s ", payload.uuid, payload)
    _LOGGER.debug(
        "%s - Time taken in inference: %.4f sec", payload.uuid, time.perf_counter() - _start_time
    )
    return orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)
