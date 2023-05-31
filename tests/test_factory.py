import unittest

from pynumaflow.function import MultiProcServer
from pynumaflow.sink import Sink

from numaprom.factory import HandlerFactory, ServerFactory

from numaprom.udf import metric_filter, Preprocess, postprocess, inference, threshold
from numaprom.udsink import train, train_rollout


class TestHandlerFactory(unittest.TestCase):
    def test_metric_filter(self):
        func = HandlerFactory.get_handler("metric_filter")
        self.assertEqual(func, metric_filter)

    def test_preprocess(self):
        obj = HandlerFactory.get_handler("preprocess")
        self.assertIsInstance(obj, Preprocess)

    def test_postprocess(self):
        func = HandlerFactory.get_handler("postprocess")
        self.assertEqual(func, postprocess)

    def test_inference(self):
        func = HandlerFactory.get_handler("inference")
        self.assertEqual(func, inference)

    def test_threshold(self):
        func = HandlerFactory.get_handler("threshold")
        self.assertEqual(func, threshold)

    def test_train(self):
        func = HandlerFactory.get_handler("train")
        self.assertEqual(func, train)

    def test_train_rollout(self):
        func = HandlerFactory.get_handler("train_rollout")
        self.assertEqual(func, train_rollout)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            HandlerFactory.get_handler("Lionel Messi")


class TestServerFactory(unittest.TestCase):
    def test_inference(self):
        func = HandlerFactory.get_handler("inference")
        obj = ServerFactory.get_server("inference", func)
        self.assertIsInstance(obj, MultiProcServer)

    def test_train(self):
        func = HandlerFactory.get_handler("train")
        obj = ServerFactory.get_server("train", func)
        self.assertIsInstance(obj, Sink)

    def test_train_rollout(self):
        func = HandlerFactory.get_handler("train_rollout")
        obj = ServerFactory.get_server("train_rollout", func)
        self.assertIsInstance(obj, Sink)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            ServerFactory.get_server("Some random string")


if __name__ == "__main__":
    unittest.main()
