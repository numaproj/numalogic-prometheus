import logging
import time
from datetime import datetime, timedelta
from typing import Tuple, Dict, List

from numalogic.models.autoencoder import AutoencoderTrainer
from numalogic.registry import ArtifactData
from numalogic.tools.data import StreamingDataset
from orjson import orjson
from pynumaflow.function import Datum
from torch.utils.data import DataLoader

from numaprom.entities import Status, StreamPayload
from numaprom.tools import (
    conditional_forward,
    load_model,
    get_metric_config,
)

_LOGGER = logging.getLogger(__name__)


def _run_model(
    payload: StreamPayload, artifact_data: ArtifactData, model_config: Dict
) -> Tuple[str, str]:
    model = artifact_data.artifact
    stream_data = payload.get_streamarray()
    stream_loader = DataLoader(StreamingDataset(stream_data, model_config["win_size"]))

    trainer = AutoencoderTrainer()
    recon_err = trainer.predict(model, dataloaders=stream_loader)

    _LOGGER.info("%s - Successfully inferred", payload.uuid)

    payload.set_win_arr(recon_err.numpy())
    payload.set_status(Status.INFERRED)
    payload.set_metadata("version", artifact_data.extras.get("version"))

    return "postprocess", orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)


@conditional_forward
def inference(_: str, datum: Datum) -> List[Tuple[str, bytes]]:

    _start_time = time.perf_counter()
    _in_msg = datum.value.decode("utf-8")
    payload = StreamPayload(**orjson.loads(_in_msg))

    _LOGGER.debug("%s - Received Payload: %r ", payload.uuid, payload)

    metric_config = get_metric_config(payload.composite_keys["name"])
    model_config = metric_config["model_config"]

    artifact_data = load_model(
        skeys=[payload.composite_keys["namespace"], payload.composite_keys["name"]],
        dkeys=[model_config["model_name"]],
    )

    train_payload = {
        "uuid": payload.uuid,
        **payload.composite_keys,
        "model_config": model_config["name"],
        "resume_training": False,
    }

    if not artifact_data:
        _LOGGER.info(
            "%s - No model found, sending to trainer. Trainer payload: %s",
            payload.uuid,
            train_payload,
        )
        return [("train", orjson.dumps(train_payload))]

    _LOGGER.debug("%s - Successfully loaded model from mlflow", payload.uuid)

    messages = []

    date_updated = artifact_data.extras["last_updated_timestamp"] / 1000
    stale_date = (
        datetime.now() - timedelta(hours=int(model_config["retrain_freq_hr"]))
    ).timestamp()

    if date_updated < stale_date:
        train_payload["resume_training"] = True
        _LOGGER.info(
            "%s - Model found is stale, sending to trainer. Trainer payload: %s",
            payload.uuid,
            train_payload,
        )
        messages.append(("train", orjson.dumps(train_payload)))

    messages.append(_run_model(payload, artifact_data, model_config))

    _LOGGER.info("%s - Sending Messages: %s ", payload.uuid, messages)
    _LOGGER.debug(
        "%s - Total time in inference: %s sec", payload.uuid, time.perf_counter() - _start_time
    )
    return messages
