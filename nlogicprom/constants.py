import os

NLOGICPROM_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.split(NLOGICPROM_DIR)[0]
TESTS_DIR = os.path.join(NLOGICPROM_DIR, "tests")
MODEL_PATH = os.path.join(ROOT_DIR, "nlogicprom/udf/models")
DATA_DIR = os.path.join(NLOGICPROM_DIR, "data")
DEFAULT_WIN_SIZE = 12
DEFAULT_THRESHOLD_MIN = 0.1
DEFAULT_MODEL_NAME = "ae_sparse"
DEFAULT_ROLLOUT_WIN_SIZE = 12
DEFAULT_ROLLOUT_THRESHOLD_MIN = 0.001
DEFAULT_ROLLOUT_MODEL_NAME = "ae_sparse"
DEFAULT_RETRAIN_FREQ_HR = 8
DEFAULT_PROMETHEUS_SERVER = "http://prometheus-service.monitoring.svc.cluster.local:8080"
CONFIG_MODEL_ID = "dataflow"
DEFAULT_RESUME_TRAINING = "True"
ARGOCD_METRICS_LIST = ["error_rate", "cpu", "error_count", "memory", "latency"]
ROLLOUTS_METRICS_LIST = ["hash_error_rate", "hash_latency"]
METRICS = [
    "namespace_app_pod_http_server_requests_errors",
    "namespace_app_pod_http_server_requests_error_rate",
    "namespace_app_pod_http_server_requests_latency",
    "namespace_asset_pod_cpu_utilization",
    "namespace_asset_pod_memory_utilization",
    "namespace_hash_pod_http_server_requests_error_rate",
    "namespace_hash_pod_http_server_requests_latency",
]
ARGO_CD = "argo_cd"
ARGO_ROLLOUTS = "argo_rollouts"
DEFAULT_TRACKING_URI = "http://mlflow-service.numalogic-prometheus.svc.cluster.local:5000"
