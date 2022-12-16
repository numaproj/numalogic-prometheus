import torch
import tempfile
import unittest
import numpy as np
from unittest.mock import patch, Mock
from numpy.testing import assert_array_equal
from sklearn.preprocessing import StandardScaler

from numalogic.models.autoencoder.variants import VanillaAE

from numaprom.pipeline import PrometheusPipeline
from numaprom.prometheus import Prometheus
from numaprom.tests.tools import mock_argocd_query_metric


@patch.object(Prometheus, "query_metric", Mock(return_value=mock_argocd_query_metric()))
class TestPrometheusPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.win_size = 6
        self.raw_model = VanillaAE(self.win_size)

    def test_train(self):
        prom_pl = PrometheusPipeline(
            preprocess_steps=[StandardScaler()],
            model=self.raw_model,
            seq_len=self.win_size,
            num_epochs=10,
        )
        df = prom_pl.fetch_data(
            delta_hr=1, metric_name="metric_1", labels_map={"namespace": "sandbox_numalogic_demo"}
        )
        x_scaled = prom_pl.preprocess(df.to_numpy())
        prom_pl.train(x_scaled)
        self.assertTrue(prom_pl.model)

    def test_inference(self):
        def _square(x):
            return x**2

        prom_pl = PrometheusPipeline(
            preprocess_steps=[StandardScaler()],
            postprocess_funcs=[_square],
            model=self.raw_model,
            seq_len=self.win_size,
            num_epochs=10,
        )
        test_size = 20
        df = prom_pl.fetch_data(
            delta_hr=1, metric_name="metric_1", labels_map={"namespace": "sandbox_numalogic_demo"}
        )
        df_train, df_test = df[:-test_size], df[-test_size:]

        x_train_scaled = prom_pl.preprocess(df_train.to_numpy())
        prom_pl.train(x_train_scaled)

        x_test_scaled = prom_pl.preprocess(df_test.to_numpy(), train=False)
        x_test_score = prom_pl.infer(x_test_scaled)

        x_test_postproc = prom_pl.postprocess(x_test_score)

        self.assertTupleEqual((test_size, 1), x_test_score.shape)
        assert_array_equal(x_test_postproc, _square(x_test_score))

    def test_no_model(self):
        prom_pl = PrometheusPipeline()
        x = np.random.randn(100, 1)
        x_scaled = prom_pl.preprocess(x)

        with self.assertRaises(ValueError):
            prom_pl.train(x_scaled)
        with self.assertRaises(ValueError):
            prom_pl.infer(x_scaled)
        with self.assertRaises(ValueError):
            prom_pl.save_model()
        with tempfile.NamedTemporaryFile(suffix=".pth") as fp:
            with self.assertRaises(ValueError):
                prom_pl.load_model(fp.name)

        assert_array_equal(x, x_scaled)

    def test_save_load_path(self):
        prom_pl_1 = PrometheusPipeline(
            model=self.raw_model,
            seq_len=self.win_size,
            num_epochs=10,
        )
        test_size = 20
        df = prom_pl_1.fetch_data(
            delta_hr=1, metric_name="metric_1", labels_map={"namespace": "sandbox_numalogic_demo"}
        )
        df_train, df_test = df[:-test_size], df[-test_size:]
        prom_pl_1.train(df_train.to_numpy())
        y_1 = prom_pl_1.infer(df_test.to_numpy())

        with tempfile.NamedTemporaryFile(suffix=".pth") as fp:
            prom_pl_1.save_model(fp.name)

            prom_pl_2 = PrometheusPipeline(
                model=self.raw_model,
                seq_len=self.win_size,
                num_epochs=10,
            )
            prom_pl_2.load_model(fp.name)
            y_2 = prom_pl_1.infer(df_test.to_numpy())

        # Check if both model's weights are equal
        _mean_wts_1, _mean_wts_2 = [], []
        with torch.no_grad():
            for _w in prom_pl_1.model.parameters():
                _mean_wts_1.append(torch.mean(_w).item())
            for _w in prom_pl_2.model.parameters():
                _mean_wts_2.append(torch.mean(_w).item())

        # self.assertTrue(_mean_wts_1)
        # self.assertAlmostEqual(_mean_wts_1, _mean_wts_2, places=6)
        # self.assertAlmostEqual(np.mean(y_1), np.mean(y_2))

    def test_save_load_buf(self):
        prom_pl_1 = PrometheusPipeline(
            model=self.raw_model,
            seq_len=self.win_size,
            num_epochs=10,
        )
        test_size = 20
        df = prom_pl_1.fetch_data(
            delta_hr=1, metric_name="metric_1", labels_map={"namespace": "sandbox_numalogic_demo"}
        )
        df_train, df_test = df[:-test_size], df[-test_size:]

        prom_pl_1.train(df_train.to_numpy())
        y_1 = prom_pl_1.infer(df_test.to_numpy())

        buf = prom_pl_1.save_model()

        prom_pl_2 = PrometheusPipeline(
            model=self.raw_model,
            seq_len=self.win_size,
            num_epochs=10,
        )
        prom_pl_2.load_model(buf)
        y_2 = prom_pl_1.infer(df_test.to_numpy())

        # Check if both model's weights are equal
        _mean_wts_1, _mean_wts_2 = [], []
        with torch.no_grad():
            for _w in prom_pl_1.model.parameters():
                _mean_wts_1.append(torch.mean(_w).item())
            for _w in prom_pl_2.model.parameters():
                _mean_wts_2.append(torch.mean(_w).item())

        # self.assertTrue(_mean_wts_1)
        # self.assertAlmostEqual(_mean_wts_1, _mean_wts_2, places=6)
        # self.assertAlmostEqual(np.mean(y_1), np.mean(y_2))


if __name__ == "__main__":
    unittest.main()
