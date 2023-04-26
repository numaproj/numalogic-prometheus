
# Metrics:

By default, `numalogic-prometheus` supports two use cases: Argo CD and Argo Rollouts. The metrics required by these
use cases are configured in [prometheus rules](../manifests/prerequisites/prometheus/prometheus-rules.yaml) and the
metrics that have `numalogic: "true"` label are collected and sent to the `numalogic-prometheus-pipeline` by the 
prometheus remote writer. 

### Argo CD:

For Argo CD use case, below are the golden signal metrics (cpu, memory, error counts, error rates and latencies).

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
For Argo CD use case, below are the golden signal metrics (error rates and latencies).
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

To on-board a new metric, add your metric to [prometheus-rules.yaml](../manifests/prerequisites/prometheus/prometheus-rules.yaml) 
and add a `numalogic: "true"` label to send it to the pipeline. 

Any new metric that is sent to the pipeline, takes in the below default config and emits the respective anomaly score.

```shell
  numalogic_config.yaml: |
    model:
      name: "SparseVanillaAE"
      conf:
        seq_len: 12
        n_features: 1
        encoder_layersizes:
          - 16
          - 8
        decoder_layersizes:
          - 8
          - 16
        dropout_p: 0.25
    trainer:
      max_epochs: 30
    preprocess:
      - name: "StandardScaler"
    threshold:
      name: "StdDevThreshold"
    postprocess:
      name: "TanhNorm"
      stateful: false
```

To configure a new metric with a specific ML parameters, provide the config in the above format. 
To get a unified anomaly score for a set of metrics, add the metrics list to `metrics` field.