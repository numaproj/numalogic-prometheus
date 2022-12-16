
# Metrics:

## Default Metrics:

By default, `numalogic-prometheus` supports two use cases: Argo CD and Argo Rollouts.

### Argo CD:

For Argo CD use case, below are the golden signal metrics (cpu, memory, error counts, error rates and latencies) configured in the [prometheus rules](https://github.com/numaproj/numalogic-prometheus/blob/main/manifests/prometheus/prometheus-rules.yaml) and sent to the pipeline.

```shell
namespace_app_pod_http_server_requests_errors
namespace_app_pod_http_server_requests_error_rate
namespace_app_pod_http_server_requests_latency
namespace_asset_pod_cpu_utilization
namespace_asset_pod_memory_utilization
```

The pipeline emits anomaly score for each individual metric and also a unified anomaly for this set of metrics.

```shell
namespace_app_pod_http_server_requests_errors_anomaly
namespace_app_pod_http_server_requests_error_rate_anomaly
namespace_app_pod_http_server_requests_latency_anomaly
namespace_asset_pod_cpu_utilization_anomaly
namespace_asset_pod_memory_utilization_anomaly
namespace_argo_cd_unified_anomaly
```

### Argo Rollouts:
For Argo CD use case, below are the golden signal metrics (error rates and latencies) configured in the [prometheus rules](../manifests/prerequisites/prometheus/prometheus-rules.yaml) and sent to the pipeline.

```shell
namespace_hash_pod_http_server_requests_error_rate
namespace_hash_pod_http_server_requests_latency
```
The pipeline emits anomaly score for each individual metric and also a unified anomaly for this set of metrics.

```shell
namespace_hash_pod_http_server_requests_error_rate_anomaly
namespace_hash_pod_http_server_requests_latency_anomaly
namespace_argo_rollouts_unified_anomaly
```

Refer to [prometheus-rules.yaml](../manifests/prerequisites/prometheus/prometheus-rules.yaml) to understand the metrics in detail.

## On-boarding New Metrics:

Any new metric that is sent to the pipeline, takes in the below default config and emits the respective anomaly score.

```shell
"default": {
    "keys": ["namespace", "name"],
    "model_config": {
        "name": "default",
        "win_size": 12,
        "threshold_min": 0.1,
        "model_name": "ae_sparse",
        "retrain_freq_hr": 8,
        "resume_training": "True",
        "num_epochs": 100,
        "keys": ["namespace", "name"],
        "scrape_interval": 5,
        "metrics": [],
    },
    "model": "VanillaAE",
}
```

To configure a new metric with a specific ML parameters, provide the config in the above format. To get a unified anomaly score for a set of metrics, add the metrics list to `metrics` field.