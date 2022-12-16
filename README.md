# numalogic-prometheus


Numalogic-prometheus is a AIOps pipeline to do in cluster anomaly detection for any prometheus metrics. It is built up on [numaflow](https://numaflow.numaproj.io/quick-start/#installation) using [nuamlogic](https://github.com/numaproj/numalogic) as the ML library.  

By default, it provides anomaly detection for Argo CD and Argo Rollouts metrics, to identify issues before deploying/rolling out new changes. 

It installs in a few minutes and is easier to onboard and configure any new metrics. 

## Prerequisites
- [Numaflow](https://numaflow.numaproj.io/quick-start/#installation)
- [Argo Workflows](https://argoproj.github.io/argo-workflows/quick-start/)
- [Prometheus](docs/prometheus.md)

## Quick Start

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


## Resources
- [PROMETHEUS](docs/prometheus.md)
- [METRICS](docs/metrics.md)
- [MLFLOW_WITH_S3](docs/mlflow-s3.md)
- [DEVELOPMENT](docs/development/development.md)
- [CONTRIBUTING](https://github.com/numaproj/numaproj/blob/main/CONTRIBUTING.md)