import pytz
import logging
import numpy as np
import pandas as pd
from torch import nn
from numpy.typing import NDArray
from sklearn.base import TransformerMixin
from sklearn.pipeline import make_pipeline
from datetime import datetime, timedelta
from typing import Sequence, Callable, Optional, Union, BinaryIO

from numalogic.models.autoencoder.factory import ModelPlFactory

from nlogicprom.constants import DEFAULT_PROMETHEUS_SERVER
from nlogicprom.prometheus import Prometheus

LOGGER = logging.getLogger(__name__)


# TODO allow saving of preprocessor functions
class PrometheusPipeline:
    """
    Class to exceute training & inference flow.
    """

    def __init__(
        self,
        namespace: str,
        metric: str,
        preprocess_steps: Sequence[TransformerMixin] = None,
        postprocess_funcs: Sequence[Callable] = None,
        model_plname="ae",
        **model_pl_kw
    ):
        self.namespace = namespace
        self.metric = metric
        self.datafetcher = None

        if model_pl_kw:
            self.model_ppl = ModelPlFactory.get_pl_obj(model_plname, **model_pl_kw)
        else:
            self.model_ppl = None

        self.preprocess_pipeline = make_pipeline(*preprocess_steps) if preprocess_steps else None
        self.postprocess_funcs = postprocess_funcs or []

    @property
    def model(self) -> Optional[nn.Module]:
        return self.model_ppl.model

    def fetch_data(
        self,
        delta_hr=36,
        end_dt: datetime = None,
        hash_col: bool = False,
        prometheus_server: str = DEFAULT_PROMETHEUS_SERVER,
    ) -> pd.DataFrame:
        self.datafetcher = Prometheus(prometheus_server)
        end_dt = end_dt or datetime.now(pytz.utc)
        start_dt = end_dt - timedelta(hours=delta_hr)

        df = self.datafetcher.query_metric(
            metric=self.metric,
            namespace=self.namespace,
            start=start_dt.timestamp(),
            end=end_dt.timestamp(),
            hash_col=hash_col,
        )
        return df

    @staticmethod
    def clean_data(df: pd.DataFrame, limit=12):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.fillna(method="ffill", limit=limit)
        df = df.fillna(method="bfill", limit=limit)
        if df.columns[df.isna().any()].tolist():
            df.dropna(inplace=True)
        return df

    # Todo: extract all good hashes, including when there are 2 hashes at a time
    def clean_rollout_data(self, df: pd.DataFrame):
        df = self.clean_data(df)
        if df.empty:
            return None
        df = df.reset_index()
        df = (
            pd.merge(df, df[df.duplicated("timestamp", keep=False)], indicator=True, how="outer")
            .query('_merge=="left_only"')
            .drop("_merge", axis=1)
        )
        df.set_index("timestamp", inplace=True)
        df.drop("hash", axis=1, inplace=True)
        df = df.sort_values(by=["timestamp"], ascending=True)
        if len(df) < (1.5 * 60 * 12):
            LOGGER.exception("Not enough training points to initiate training")
            return None
        return df

    def preprocess(self, X: NDArray, train=True) -> NDArray[float]:
        if not self.preprocess_pipeline:
            LOGGER.warning("No preprocess steps provided.")
            return X
        if train:
            return self.preprocess_pipeline.fit_transform(X)
        return self.preprocess_pipeline.transform(X)

    def train(self, X: NDArray) -> None:
        """
        Infer/predict on the given data.
        Note: this assumes that X is already preprocessed.
        :param X: Numpy Array
        """
        if not self.model_ppl:
            raise ValueError("Model pipeline is not initialized.")
        self.model_ppl.fit(X)

    def infer(self, X: NDArray) -> NDArray[float]:
        """
        Infer/predict on the given data.
        Note: this assumes that X is already preprocessed.
        :param X: Numpy Array
        :return: Anomaly scores
        """
        if not self.model_ppl:
            raise ValueError("Model pipeline is not initialized.")
        return self.model_ppl.score(X)

    def postprocess(self, y: NDArray) -> NDArray[float]:
        for func in self.postprocess_funcs:
            y = func(np.copy(y))
        return y

    def save_model(self, path: Union[str, None] = None) -> Optional[BinaryIO]:
        if not self.model_ppl:
            raise ValueError("Model pipeline is not initialized.")
        return self.model_ppl.save(path)

    def load_model(
        self, path_or_buf: Union[str, BinaryIO] = None, model: nn.Module = None, **metadata
    ) -> None:
        if not self.model_ppl:
            raise ValueError(
                "An initialized model pipeline object is required for loading a saved model."
            )
        self.model_ppl.load(path=path_or_buf, model=model, **metadata)
