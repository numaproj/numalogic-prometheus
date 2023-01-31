import os

NUMAPROM_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.split(NUMAPROM_DIR)[0]
TESTS_DIR = os.path.join(ROOT_DIR, "tests")
DATA_DIR = os.path.join(NUMAPROM_DIR, "data")

# endpoints
DEFAULT_PROMETHEUS_SERVER = "http://prometheus-service.monitoring.svc.cluster.local:8080"
DEFAULT_TRACKING_URI = "http://mlflow-service.numalogic-prometheus.svc.cluster.local:5000"


# UDF constants
TRAIN_VTX_KEY = "train"
INFERENCE_VTX_KEY = "inference"
THRESHOLD_VTX_KEY = "threshold"
POSTPROC_VTX_KEY = "postproc"


# ML parameters
MODEL_CONFIG = {
    "argo_cd": {
        "name": "argo_cd",
        "win_size": 12,
        "threshold_min": 0.1,
        "model_name": "ae_sparse",
        "retrain_freq_hr": 8,
        "resume_training": "True",
        "num_epochs": 100,
        "training_keys": ["namespace", "name"],
        "scrape_interval": 5,
        "unified_anomaly": "namespace_app_pod_http_server_requests_unified_anomaly",
        "metrics": [
            "namespace_app_pod_http_server_requests_errors",
            "namespace_app_pod_http_server_requests_error_rate",
            "namespace_app_pod_http_server_requests_latency",
            "namespace_asset_pod_cpu_utilization",
            "namespace_asset_pod_memory_utilization",
        ],
    },
    "argo_rollouts": {
        "name": "argo_rollouts",
        "win_size": 12,
        "threshold_min": 0.001,
        "model_name": "ae_sparse",
        "retrain_freq_hr": 8,
        "resume_training": "True",
        "num_epochs": 50,
        "training_keys": ["namespace", "name"],
        "scrape_interval": 5,
        "unified_anomaly": "namespace_hash_pod_http_server_requests_unified_anomaly",
        "metrics": [
            "namespace_hash_pod_http_server_requests_error_rate",
            "namespace_hash_pod_http_server_requests_latency",
        ],
    },
    "fuzzy_argocd_qal": {
        "name": "fuzzy_argocd_qal",
        "win_size": 12,
        "threshold_min": 0.1,
        "model_name": "ae_sparse",
        "retrain_freq_hr": 8,
        "resume_training": "True",
        "num_epochs": 100,
        "training_keys": ["namespace", "name"],
        "scrape_interval": 5,
        "unified_anomaly": "namespace_argocd_o11yfuzzygqlfederation_segment_api_unified_anomaly_qal",
        "metrics": [
            "namespace_argocd_o11yfuzzygqlfederation_segment_api_latency_qal",
            "namespace_argocd_o11yfuzzygqlfederation_segment_api_error_count_qal",
        ],
    },
    "fuzzy_argocd_ppd": {
        "name": "fuzzy_argocd_ppd",
        "win_size": 12,
        "threshold_min": 0.1,
        "model_name": "ae_sparse",
        "retrain_freq_hr": 8,
        "resume_training": "True",
        "num_epochs": 100,
        "training_keys": ["namespace", "name"],
        "scrape_interval": 5,
        "unified_anomaly": "namespace_argocd_o11yfuzzygqlfederation_segment_api_unified_anomaly_ppd",
        "metrics": [
            "namespace_argocd_o11yfuzzygqlfederation_segment_api_latency_ppd",
            "namespace_argocd_o11yfuzzygqlfederation_segment_api_error_count_ppd",
        ],
    },
    "fuzzy_rollouts_qal": {
        "name": "fuzzy_rollouts",
        "win_size": 12,
        "threshold_min": 0.001,
        "model_name": "ae_sparse",
        "retrain_freq_hr": 8,
        "resume_training": "True",
        "num_epochs": 50,
        "training_keys": ["namespace", "name"],
        "scrape_interval": 5,
        "unified_anomaly": "namespace_rollout_o11yfuzzygqlfederation_segment_api_unified_anomaly_qal",
        "metrics": [
            "namespace_rollout_o11yfuzzygqlfederation_segment_api_latency_qal",
            "namespace_rollout_o11yfuzzygqlfederation_segment_api_error_count_qal",
        ],
    },
    "fuzzy_rollouts_ppd": {
        "name": "fuzzy_rollouts_ppd",
        "win_size": 12,
        "threshold_min": 0.001,
        "model_name": "ae_sparse",
        "retrain_freq_hr": 8,
        "resume_training": "True",
        "num_epochs": 50,
        "training_keys": ["namespace", "name"],
        "scrape_interval": 5,
        "unified_anomaly": "namespace_rollout_o11yfuzzygqlfederation_segment_api_unified_anomaly_ppd",
        "metrics": [
            "namespace_rollout_o11yfuzzygqlfederation_segment_api_latency_ppd",
            "namespace_rollout_o11yfuzzygqlfederation_segment_api_error_count_ppd",
        ],
    },
    "default": {
        "name": "default",
        "win_size": 12,
        "threshold_min": 0.1,
        "model_name": "ae_sparse",
        "retrain_freq_hr": 8,
        "resume_training": "True",
        "num_epochs": 100,
        "training_keys": ["namespace", "name"],
        "scrape_interval": 5,
        "unified_anomaly": None,
        "metrics": [],
    },
}

METRIC_CONFIG = {
    "namespace_app_pod_http_server_requests_errors": {
        "keys": ["namespace", "name"],
        "model_config": MODEL_CONFIG["argo_cd"],
        "model": "VanillaAE",
    },
    "namespace_app_pod_http_server_requests_error_rate": {
        "keys": ["namespace", "name"],
        "model_config": MODEL_CONFIG["argo_cd"],
        "model": "VanillaAE",
    },
    "namespace_app_pod_http_server_requests_latency": {
        "keys": ["namespace", "name"],
        "model_config": MODEL_CONFIG["argo_cd"],
        "model": "Conv1dAE",
    },
    "namespace_asset_pod_cpu_utilization": {
        "keys": ["namespace", "name"],
        "model_config": MODEL_CONFIG["argo_cd"],
        "model": "VanillaAE",
    },
    "namespace_asset_pod_memory_utilization": {
        "keys": ["namespace", "name"],
        "model_config": MODEL_CONFIG["argo_cd"],
        "model": "VanillaAE",
    },
    "namespace_hash_pod_http_server_requests_error_rate": {
        "keys": ["namespace", "name", "hash_id"],
        "model_config": MODEL_CONFIG["argo_rollouts"],
        "model": "VanillaAE",
    },
    "namespace_hash_pod_http_server_requests_latency": {
        "keys": ["namespace", "name", "hash_id"],
        "model_config": MODEL_CONFIG["argo_rollouts"],
        "model": "VanillaAE",
    },
    "namespace_argocd_o11yfuzzygqlfederation_segment_api_latency_qal": {
        "keys": ["namespace", "name"],
        "model_config": MODEL_CONFIG["fuzzy_argocd_qal"],
        "model": "VanillaAE",
    },
    "namespace_argocd_o11yfuzzygqlfederation_segment_api_error_count_qal": {
        "keys": ["namespace", "name"],
        "model_config": MODEL_CONFIG["fuzzy_argocd_qal"],
        "model": "VanillaAE",
    },
    "namespace_rollout_o11yfuzzygqlfederation_segment_api_latency_qal": {
        "keys": ["namespace", "name", "hash_id"],
        "model_config": MODEL_CONFIG["fuzzy_rollouts_qal"],
        "model": "VanillaAE",
    },
    "namespace_rollout_o11yfuzzygqlfederation_segment_api_error_count_qal": {
        "keys": ["namespace", "name", "hash_id"],
        "model_config": MODEL_CONFIG["fuzzy_rollouts_qal"],
        "model": "VanillaAE",
    },
    "namespace_argocd_o11yfuzzygqlfederation_segment_api_latency_ppd": {
        "keys": ["namespace", "name"],
        "model_config": MODEL_CONFIG["fuzzy_argocd_ppd"],
        "model": "VanillaAE",
    },
    "namespace_argocd_o11yfuzzygqlfederation_segment_api_error_count_ppd": {
        "keys": ["namespace", "name"],
        "model_config": MODEL_CONFIG["fuzzy_argocd_ppd"],
        "model": "VanillaAE",
    },
    "namespace_rollout_o11yfuzzygqlfederation_segment_api_latency_ppd": {
        "keys": ["namespace", "name", "hash_id"],
        "model_config": MODEL_CONFIG["fuzzy_rollouts_ppd"],
        "model": "VanillaAE",
    },
    "namespace_rollout_o11yfuzzygqlfederation_segment_api_error_count_ppd": {
        "keys": ["namespace", "name", "hash_id"],
        "model_config": MODEL_CONFIG["fuzzy_rollouts_ppd"],
        "model": "VanillaAE",
    },
    "default": {
        "keys": ["namespace", "name", "hash_id"],
        "model_config": MODEL_CONFIG["default"],
        "model": "VanillaAE",
    },
}
