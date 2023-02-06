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


OUTPUT_CONFIG = {
    "argo_cd": {
        "unified_strategy": "max",
        "unified_metric_name": "namespace_app_pod_http_server_requests_unified_anomaly",
        "unified_metrics": [
            "namespace_app_pod_http_server_requests_errors",
            "namespace_app_pod_http_server_requests_error_rate",
            "namespace_app_pod_http_server_requests_latency",
            "namespace_asset_pod_cpu_utilization",
            "namespace_asset_pod_memory_utilization",
        ],
    },
    "argo_rollouts": {
        "unified_strategy": "max",
        "unified_metric_name": "namespace_hash_pod_http_server_requests_unified_anomaly",
        "unified_metrics": [
            "namespace_hash_pod_http_server_requests_error_rate",
            "namespace_hash_pod_http_server_requests_latency",
        ],
    },
    "fuzzy_rollouts_qal": {
        "unified_strategy": "max",
        "unified_metric_name": "namespace_rollout_o11yfuzzygqlfederation_segment_api_unified_anomaly_qal",
        "unified_metrics": [
            "namespace_rollout_o11yfuzzygqlfederation_segment_api_latency_qal",
            "namespace_argocd_o11yfuzzygqlfederation_segment_api_error_rate_qal",
        ],
    },
    "fuzzy_rollouts_ppd": {
        "unified_strategy": "max",
        "unified_metric_name": "namespace_rollout_o11yfuzzygqlfederation_segment_api_unified_anomaly_ppd",
        "unified_metrics": [
            "namespace_rollout_o11yfuzzygqlfederation_segment_api_latency_ppd",
            "namespace_argocd_o11yfuzzygqlfederation_segment_api_error_rate_ppd",
        ],
    },
    "fuzzy_rollouts_stg": {
        "unified_strategy": "max",
        "unified_metric_name": "namespace_rollout_o11yfuzzygqlfederation_segment_api_unified_anomaly_stg",
        "unified_metrics": [
            "namespace_rollout_o11yfuzzygqlfederation_segment_api_latency_stg",
            "namespace_argocd_o11yfuzzygqlfederation_segment_api_error_rate_stg",
        ],
    },
    "fuzzy_rollouts_prd": {
        "unified_strategy": "max",
        "unified_metric_name": "namespace_rollout_o11yfuzzygqlfederation_segment_api_unified_anomaly_prd",
        "unified_metrics": [
            "namespace_rollout_o11yfuzzygqlfederation_segment_api_latency_prd",
            "namespace_argocd_o11yfuzzygqlfederation_segment_api_error_rate_prd",
        ],
    },
    "default": {"unified_strategy": None, "unified_metric_name": None, "unified_metrics": None},
}

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
    },
    "fuzzy_rollouts": {
        "name": "fuzzy_rollouts",
        "win_size": 12,
        "threshold_min": 0.001,
        "model_name": "ae_sparse",
        "retrain_freq_hr": 8,
        "resume_training": "True",
        "num_epochs": 50,
        "training_keys": ["namespace", "name"],
    },
    "default": {
        "name": "default",
        "win_size": 12,
        "threshold_min": 0.001,
        "model_name": "ae_sparse",
        "retrain_freq_hr": 8,
        "resume_training": "True",
        "num_epochs": 50,
        "training_keys": ["namespace", "name"],
    },
}

METRIC_CONFIG = {
    "namespace_app_pod_http_server_requests_errors": {
        "keys": ["namespace", "name"],
        "scrape_interval": 5,
        "model_config": MODEL_CONFIG["argo_cd"],
        "output_config": OUTPUT_CONFIG["argo_cd"],
    },
    "namespace_app_pod_http_server_requests_error_rate": {
        "keys": ["namespace", "name"],
        "scrape_interval": 5,
        "model_config": MODEL_CONFIG["argo_cd"],
        "output_config": OUTPUT_CONFIG["argo_cd"],
    },
    "namespace_app_pod_http_server_requests_latency": {
        "keys": ["namespace", "name"],
        "scrape_interval": 5,
        "model_config": MODEL_CONFIG["argo_cd"],
        "output_config": OUTPUT_CONFIG["argo_cd"],
    },
    "namespace_asset_pod_cpu_utilization": {
        "keys": ["namespace", "name"],
        "scrape_interval": 5,
        "model_config": MODEL_CONFIG["argo_cd"],
        "output_config": OUTPUT_CONFIG["argo_cd"],
    },
    "namespace_asset_pod_memory_utilization": {
        "keys": ["namespace", "name"],
        "scrape_interval": 5,
        "model_config": MODEL_CONFIG["argo_cd"],
        "output_config": OUTPUT_CONFIG["argo_cd"],
    },
    "namespace_hash_pod_http_server_requests_error_rate": {
        "keys": ["namespace", "name", "hash_id"],
        "scrape_interval": 5,
        "model_config": MODEL_CONFIG["argo_rollouts"],
        "output_config": OUTPUT_CONFIG["argo_rollouts"],
    },
    "namespace_hash_pod_http_server_requests_latency": {
        "keys": ["namespace", "name", "hash_id"],
        "scrape_interval": 5,
        "model_config": MODEL_CONFIG["argo_rollouts"],
        "output_config": OUTPUT_CONFIG["argo_rollouts"],
    },
    "namespace_rollout_o11yfuzzygqlfederation_segment_api_latency_qal": {
        "keys": ["namespace", "name", "hash_id"],
        "scrape_interval": 30,
        "model_config": MODEL_CONFIG["fuzzy_rollouts"],
        "output_config": OUTPUT_CONFIG["fuzzy_rollouts_qal"],
    },
    "namespace_argocd_o11yfuzzygqlfederation_segment_api_error_rate_qal": {
        "keys": ["namespace", "name", "hash_id"],
        "scrape_interval": 30,
        "model_config": MODEL_CONFIG["fuzzy_rollouts"],
        "output_config": OUTPUT_CONFIG["fuzzy_rollouts_qal"],
    },
    "namespace_rollout_o11yfuzzygqlfederation_segment_api_latency_ppd": {
        "keys": ["namespace", "name", "hash_id"],
        "scrape_interval": 30,
        "model_config": MODEL_CONFIG["fuzzy_rollouts"],
        "output_config": OUTPUT_CONFIG["fuzzy_rollouts_ppd"],
    },
    "namespace_argocd_o11yfuzzygqlfederation_segment_api_error_rate_ppd": {
        "keys": ["namespace", "name", "hash_id"],
        "scrape_interval": 30,
        "model_config": MODEL_CONFIG["fuzzy_rollouts"],
        "output_config": OUTPUT_CONFIG["fuzzy_rollouts_ppd"],
    },
    "namespace_rollout_o11yfuzzygqlfederation_segment_api_latency_stg": {
        "keys": ["namespace", "name", "hash_id"],
        "scrape_interval": 30,
        "model_config": MODEL_CONFIG["fuzzy_rollouts"],
        "output_config": OUTPUT_CONFIG["fuzzy_rollouts_stg"],
    },
    "namespace_argocd_o11yfuzzygqlfederation_segment_api_error_rate_stg": {
        "keys": ["namespace", "name", "hash_id"],
        "scrape_interval": 30,
        "model_config": MODEL_CONFIG["fuzzy_rollouts"],
        "output_config": OUTPUT_CONFIG["fuzzy_rollouts_stg"],
    },
    "namespace_rollout_o11yfuzzygqlfederation_segment_api_latency_prd": {
        "keys": ["namespace", "name", "hash_id"],
        "scrape_interval": 30,
        "model_config": MODEL_CONFIG["fuzzy_rollouts"],
        "output_config": OUTPUT_CONFIG["fuzzy_rollouts_prd"],
    },
    "namespace_argocd_o11yfuzzygqlfederation_segment_api_error_rate_prd": {
        "keys": ["namespace", "name", "hash_id"],
        "scrape_interval": 30,
        "model_config": MODEL_CONFIG["fuzzy_rollouts"],
        "output_config": OUTPUT_CONFIG["fuzzy_rollouts_prd"],
    },
    "default": {
        "keys": ["namespace", "name", "hash_id"],
        "scrape_interval": 30,
        "model_config": MODEL_CONFIG["default"],
        "output_config": OUTPUT_CONFIG["default"],
    },
}
