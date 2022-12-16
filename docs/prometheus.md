# Prometheus Setup:


## Installation

Run the below command to install prometheus and its dependencies in you cluster.

```shell
kustomize build manifests/prometheus/ | kubectl apply -f - 
```

## Remote Writer:

Prometheus remote write config, that filters the metrics by its `__name__`, `numalogic` flag and writes to the input vertex of the `numalogic-prometheus-pipeline`.

```shell
remote_write:
      -
        name: prometheus
        url: "https://numalogic-prometheus-pipeline-input.numalogic-prometheus.svc.cluster.local:8443/vertices/input"
        remote_timeout: 1m
        queue_config:
          capacity: 10000
          min_shards: 10
          max_shards: 100
          max_samples_per_send: 1000
          batch_send_deadline: 10s
          min_backoff: 30ms
          max_backoff: 100ms
        tls_config:
          insecure_skip_verify: true
        write_relabel_configs:
        - action: keep
          regex: namespace_*;true
          source_labels:
          - __name__
          - numalogic
```