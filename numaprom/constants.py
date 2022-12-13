import os

NUMAPROM_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.split(NUMAPROM_DIR)[0]
TESTS_DIR = os.path.join(NUMAPROM_DIR, "tests")
MODEL_PATH = os.path.join(ROOT_DIR, "numaprom/udf/models")
DATA_DIR = os.path.join(NUMAPROM_DIR, "data")

CONFIG_MODEL_ID = "dataflow"
ARGO_CD = "argo_cd"
ARGO_ROLLOUTS = "argo_rollouts"

# endpoints
DEFAULT_PROMETHEUS_SERVER = "http://prometheus-service.monitoring.svc.cluster.local:8080"
DEFAULT_TRACKING_URI = "http://mlflow-service.numalogic-prometheus.svc.cluster.local:5000"

# ML parameters
MODEL_CONFIG = {
    ARGO_CD: {
        "name": ARGO_CD,
        "win_size": 12,
        "threshold_min": 0.1,
        "model_name": "ae_sparse",
        "retrain_freq_hr": 8,
        "resume_training": "True",
        "keys": ["namespace", "name"],
        "metrics": [
            "namespace_app_pod_http_server_requests_errors",
            "namespace_app_pod_http_server_requests_error_rate",
            "namespace_app_pod_http_server_requests_latency",
            "namespace_asset_pod_cpu_utilization",
            "namespace_asset_pod_memory_utilization"
        ],
    },
    ARGO_ROLLOUTS: {
        "name": ARGO_ROLLOUTS,
        "win_size": 12,
        "threshold_min": 0.001,
        "model_name": "ae_sparse",
        "retrain_freq_hr": 8,
        "resume_training": "True",
        "keys": ["namespace", "name"],
        "metrics": [
            "namespace_hash_pod_http_server_requests_error_rate",
            "namespace_hash_pod_http_server_requests_latency",
        ]
    }
}

METRIC_CONFIG = {
    "namespace_app_pod_http_server_requests_errors": {
        "keys": ["namespace", "name"],
        "model_config": MODEL_CONFIG[ARGO_CD]

    },
    "namespace_app_pod_http_server_requests_error_rate": {
        "keys": ["namespace", "name"],
        "model_config": MODEL_CONFIG[ARGO_CD]
    },
    "namespace_app_pod_http_server_requests_latency": {
        "keys": ["namespace", "name"],
        "model_config": MODEL_CONFIG[ARGO_CD]
    },
    "namespace_asset_pod_cpu_utilization": {
        "keys": ["namespace", "name"],
        "model_config": MODEL_CONFIG[ARGO_CD]
    },
    "namespace_asset_pod_memory_utilization": {
        "keys": ["namespace", "name"],
        "model_config": MODEL_CONFIG[ARGO_CD]
    },
    "namespace_hash_pod_http_server_requests_error_rate": {
        "keys": ["namespace", "name", "hash_id"],
        "model_config": MODEL_CONFIG[ARGO_ROLLOUTS]
    },
    "namespace_hash_pod_http_server_requests_latency": {
        "keys": ["namespace", "name", "hash_id"],
        "model_config":  MODEL_CONFIG[ARGO_ROLLOUTS]
    }
}

