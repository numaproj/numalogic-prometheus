import os

NUMAPROM_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.split(NUMAPROM_DIR)[0]
TESTS_DIR = os.path.join(ROOT_DIR, "tests")
TESTS_RESOURCES = os.path.join(TESTS_DIR, "resources")
DATA_DIR = os.path.join(NUMAPROM_DIR, "data")
CONFIG_DIR = os.path.join(NUMAPROM_DIR, "configs")
DEFAULT_CONFIG_DIR = os.path.join(NUMAPROM_DIR, "default-configs")
