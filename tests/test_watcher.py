import unittest
from unittest.mock import patch, Mock

from numaprom import watcher
from numaprom.watcher import update_configs, load_configs
from tests.tools import mock_configs


@patch.object(watcher, "load_configs", Mock(return_value=mock_configs()))
class TestWatcher(unittest.TestCase):
    def test_update_configs(self):
        config = update_configs()
        self.assertTrue(len(config), 3)

    def test_load_configs(self):
        app_configs, default_configs, default_numalogic = load_configs()
        print(type(app_configs))
        print(type(default_configs))
