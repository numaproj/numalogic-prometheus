import unittest

from numaprom.factory import HandlerFactory

from numaprom.udf import metric_filter, preprocess, postprocess, inference, threshold
from numaprom.udsink import train, train_rollout


class TestFactory(unittest.TestCase):
    def test_metric_filter(self):
        func = HandlerFactory.get_handler("metric_filter")
        self.assertEqual(func, metric_filter)

    def test_preprocess(self):
        func = HandlerFactory.get_handler("preprocess")
        self.assertEqual(func, preprocess)

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
        with self.assertRaises(NotImplementedError):
            HandlerFactory.get_handler("Lionel Messi")


if __name__ == "__main__":
    unittest.main()
