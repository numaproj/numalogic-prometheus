import logging
import os
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Dict, Optional

from numalogic.models.autoencoder import AutoencoderTrainer
from numalogic.registry import ArtifactData
from numalogic.tools.data import StreamingDataset
from orjson import orjson
from pynumaflow.function import Datum
from torch.utils.data import DataLoader

from numaprom.entities import Status, StreamPayload, TrainerPayload, Header
from numaprom.entities import PayloadFactory
from numaprom.tools import (
    load_model,
    get_metric_config,
    msgs_forward,
    calculate_static_thresh,
)

_LOGGER = logging.getLogger(__name__)
STATIC_LIMIT: float = float(os.getenv("STATIC_LIMIT", 3.0))


def _run_inference(
    payload: StreamPayload, artifact_data: ArtifactData, model_config: Dict
) -> StreamPayload:
    model = artifact_data.artifact
    stream_data = payload.get_stream_array()
    stream_loader = DataLoader(StreamingDataset(stream_data, model_config["win_size"]))

    trainer = AutoencoderTrainer()
    recon_err = trainer.predict(model, dataloaders=stream_loader)

    _LOGGER.info("%s - Successfully inferred", payload.uuid)

    payload.set_win_arr(recon_err.numpy())
    payload.set_status(Status.INFERRED)
    payload.set_metadata("version", artifact_data.extras.get("version"))
    return payload


def _get_model(payload: StreamPayload, model_config: dict) -> Optional[ArtifactData]:
    artifact_data = load_model(
        skeys=[payload.composite_keys["namespace"], payload.composite_keys["name"]],
        dkeys=[model_config["model_name"]],
    )
    if not artifact_data:
        payload.set_status(Status.ARTIFACT_NOT_FOUND)
        _LOGGER.info(
            "%s - Model not found for %s",
            payload.uuid,
            payload.composite_keys,
        )
        return None
    _LOGGER.debug("%s - Successfully loaded model from mlflow", payload.uuid)
    return artifact_data


def _is_model_stale(
    payload: StreamPayload, artifact_data: ArtifactData, model_config: dict
) -> bool:
    date_updated = artifact_data.extras["last_updated_timestamp"] / 1000
    stale_date = (
        datetime.now() - timedelta(hours=int(model_config["retrain_freq_hr"]))
    ).timestamp()
    if date_updated < stale_date:
        _LOGGER.info(
            "%s - Model found is stale for %s",
            payload.uuid,
            payload.composite_keys,
        )
        return True
    return False


@msgs_forward
def inference(_: str, datum: Datum) -> list[bytes]:
    _start_time = time.perf_counter()

    _in_msg = datum.value.decode("utf-8")

    # Construct payload object
    payload = PayloadFactory.from_json(_in_msg)

    # Check if trainer payload is passed on from previous vtx
    if isinstance(payload, TrainerPayload):
        _LOGGER.debug("%s - Relaying forward trainer payload")
        return [orjson.dumps(payload)]

    # Check if this payload has performed static thresholding
    if payload.header == Header.STATIC_INFERENCE:
        _LOGGER.debug("%s - Relaying forward static threshold payload")
        return [orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)]

    messages = []

    # Load config
    metric_config = get_metric_config(payload.composite_keys["name"])
    model_config = metric_config["model_config"]

    # Check if model exists
    artifact_data = _get_model(payload, model_config)
    if not artifact_data:
        msgs = []
        train_payload = TrainerPayload(
            uuid=payload.uuid, composite_keys=OrderedDict(payload.composite_keys)
        )
        msgs.append(orjson.dumps(train_payload))

        # Calculate scores using static threshold
        msgs.append(calculate_static_thresh(payload, STATIC_LIMIT))
        return msgs

    # Check if current model is stale
    if _is_model_stale(payload, artifact_data, model_config):
        train_payload = TrainerPayload(
            uuid=payload.uuid, composite_keys=OrderedDict(payload.composite_keys)
        )
        messages.append(orjson.dumps(train_payload))

    # Generate predictions
    payload = _run_inference(payload, artifact_data, model_config)
    messages.append(orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY))

    _LOGGER.info("%s - Sending Payload: %s ", payload.uuid, payload)
    _LOGGER.debug(
        "%s - Time taken in inference: %.4f sec", payload.uuid, time.perf_counter() - _start_time
    )
    return messages
