import os
import unittest

from nlogicprom.constants import TESTS_DIR
from nlogicprom.tests.tools import get_datum
from nlogicprom.tools import get_data
from nlogicprom.udf import metric_filter

DATA_DIR = os.path.join(TESTS_DIR, "resources", "data")
LATENCY = os.path.join(DATA_DIR, "latency.txt")
REQ_2xx = os.path.join(DATA_DIR, "2xx.txt")


class TestFilter(unittest.TestCase):
    LATENCY = None
    REQ_2xx = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.LATENCY = get_datum(get_data(LATENCY))
        cls.REQ_2xx = get_datum(get_data(REQ_2xx))

    def test_filter1(self):
        _out = metric_filter(None, self.LATENCY)
        data = _out._messages[0]._value.decode("utf-8")
        self.assertTrue(data)

    def test_filter2(self):
        _out = metric_filter(None, self.REQ_2xx)
        data = _out._messages[0]._value.decode("utf-8")
        self.assertFalse(data)


if __name__ == "__main__":
    unittest.main()
