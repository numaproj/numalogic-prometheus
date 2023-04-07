import unittest
from unittest.mock import patch, Mock

from numaprom import watcher
from numaprom.watcher import update_configs
from tests.tools import mock_configs


@patch.object(watcher, "update_configs", Mock(return_value=mock_configs()))
class TestWatcher(unittest.TestCase):
    def test_update_configs(self):
        config = update_configs()
        self.assertTrue(len(config), 3)
