import unittest

from nlogicprom.factory import HandlerFactory

from nlogicprom.udf import metric_filter, preprocess, postprocess, inference


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

    def test_invalid(self):
        with self.assertRaises(NotImplementedError):
            HandlerFactory.get_handler("Lionel Messi")


if __name__ == "__main__":
    unittest.main()
