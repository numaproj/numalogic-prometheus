import logging
import time

import pandas as pd
from numalogic.preprocess.transformer import LogTransformer
from pynumaflow.function import Messages, Datum

from nlogicprom.entities import Status, Payload
from nlogicprom.pipeline import PrometheusPipeline
from nlogicprom.tools import catch_exception, msg_forward, get_metrics

LOGGER = logging.getLogger(__name__)


@catch_exception
@msg_forward
def preprocess(key: str, datum: Datum) -> Messages:
    start_preprocess = time.time()
    payload = Payload.from_json(datum.value.decode("utf-8"))

    pipeline = PrometheusPipeline(
        namespace=payload.namespace,
        metric=payload.metric,
        preprocess_steps=[LogTransformer()],
    )

    df = payload.get_processed_dataframe()
    arr = pipeline.preprocess(df.to_numpy(), train=False)
    df = pd.DataFrame(data=arr, columns=df.columns, index=df.index).reset_index()
    payload.processedMetrics = get_metrics(df)
    payload.status = Status.PRE_PROCESSED

    payload_json = payload.to_json()
    LOGGER.info("%s - Successfully pre-processed payload: %s", payload.uuid, payload_json)
    LOGGER.info(
        "%s - Total time to preprocess: %s",
        payload.uuid,
        time.time() - start_preprocess,
    )

    return payload_json
