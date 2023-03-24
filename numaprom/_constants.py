import os

NUMAPROM_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.split(NUMAPROM_DIR)[0]
TESTS_DIR = os.path.join(ROOT_DIR, "tests")
DATA_DIR = os.path.join(NUMAPROM_DIR, "data")
CONFIG_DIR = os.path.join(NUMAPROM_DIR, "configs")
DEFAULT_CONFIG_DIR = os.path.join(NUMAPROM_DIR, "default-configs")

# endpoints
DEFAULT_PROMETHEUS_SERVER = "http://prometheus-service.monitoring.svc.cluster.local:8080"
DEFAULT_TRACKING_URI = "http://mlflow-service.numalogic-prometheus.svc.cluster.local:5000"


# UDF constants
TRAIN_VTX_KEY = "train"
INFERENCE_VTX_KEY = "inference"
THRESHOLD_VTX_KEY = "threshold"
POSTPROC_VTX_KEY = "postproc"
