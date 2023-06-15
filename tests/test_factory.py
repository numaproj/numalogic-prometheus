import unittest

from numaprom.factory import HandlerFactory

class TestFactory(unittest.TestCase):
    def test_metric_filter(self):
        func = HandlerFactory.get_handler("window")
        self.assertTrue(func)

    def test_invalid(self):
        with self.assertRaises(NotImplementedError):
            HandlerFactory.get_handler("Lionel Messi")


if __name__ == "__main__":
    unittest.main()
