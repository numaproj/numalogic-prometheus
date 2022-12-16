# numalogic-prometheus

## Prerequisites
- [Numaflow](https://numaflow.numaproj.io/quick-start/#installation)
- [Prometheus](https://prometheus.io/docs/prometheus/latest/getting_started/)
- [Argo Workflows](https://argoproj.github.io/argo-workflows/quick-start/)


## Quick Start:

Run the below command to install `numalogic-prometheus` and its dependencies in you cluster.

```shell
kustomize build numalogic-prometheus/manifests/ | kubectl apply -f - 
```


## Operations

- To view Numaflow UX:
```
kubectl port-forward svc/numaflow-server 8443  -n numaflow-system
```

Here you can see all the pipelines running on the cluster

- To view ML flow server:
```
kubectl port-forward svc/mlflow-service 5000 -n numalogic-prometheus
```

Here you can see all the model runs, models saved, etc.

- To view Prometheus server:
```
kubectl port-forward <prometheus-deployment-xxxxxxxxx-xxxxx> 8490:9090 -n monitoring
```



