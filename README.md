# numalogic-prometheus

## Prerequisites
- [Numaflow](https://numaflow.numaproj.io/quick-start/#installation)
- [Prometheus](https://prometheus.io/docs/prometheus/latest/getting_started/)
- [AWS](https://aws.amazon.com/)

## Instructions

- Fork and clone the repository to your local and build numalogic-prometheus docker image.

```shell
git@github.com:numaproj/numalogic-prometheus.git

docker build -t numalogic-prometheus . && k3d image import docker.io/library/numalogic-prometheus
```

- Create numalogic-prometheus namespace.
```shell
kubectl create namespace numalogic-prometheus

kubectl config set-context --current --namespace=numalogic-prometheus
```


## Redis Setup
Install redis in `numalogic-prometheus` namespace and copy the password to notepad.

```shell
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install numalogic bitnami/redis-cluster
echo $(kubectl get secret --namespace "numalogic-prometheus" numalogic-redis-cluster -o jsonpath="{.data.redis-password}" | base64 -d)
```

## MLflow Setup

### AWS S3 Bucket Setup:
- Create an [AWS S3 bucket](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html)
- Create an [AWS IAM Role](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_create.html), with the following policy. 
 
  Note: Replace the `BUKET_NAME` with your bucket name.
```shell
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "VisualEditor0",
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:GetObjectAcl",
                "s3:GetObject",
                "s3:DeleteObjectVersion",
                "s3:ListBucket",
                "s3:DeleteObject",
                "s3:PutObjectAcl",
                "s3:GetObjectVersion"
            ],
            "Resource": [
                "arn:aws:s3:::BUCKET_NAME",
                "arn:aws:s3:::BUCKET_NAME/*",
            ]
        }
    ]
}
```
- Attach the created role to the s3 bucket under permissions.

Note: Replace the `BUKET_NAME`, `ACCOUNT_NUMBER`, `ROLE_NAME` with your S3 bucket name, AWS account ID, and the AWS role name.
```shell
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AddCannedAcl",
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::ACCOUNT_ID:role/ROLE_NAME>"
            },
            "Action": [
                "s3:PutObject",
                "s3:GetObjectAcl",
                "s3:GetObject",
                "s3:DeleteObjectVersion",
                "s3:ListBucket",
                "s3:DeleteObject",
                "s3:PutObjectAcl",
                "s3:GetObjectVersion"
            ],
            "Resource": [
                "arn:aws:s3:::BUCKET_NAME",
                "arn:aws:s3:::BUCKET_NAME/*"
            ]
        }
    ]
}
```
- Create an [AWS IAM User](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_users_create.html#id_users_create_console) for the role created, by adding the role under permissions.

### MLflow Server Setup:
Build Mlflow docker image and install MLflow server in the `numalogic-prometheus` namespace.

Note: 
 1. Replace [MLFLOW_S3_USER_ARN](https://github.com/numaproj/numalogic-prometheus/blob/main/deployment/mlflow-deployment.yaml#L14) in mlflow-deployment.yaml with your AWS User arn. 
 2. Replace [MLFLOW_BUCKET](https://github.com/numaproj/numalogic-prometheus/blob/main/deployment/mlflow-deployment.yaml#L28) in mlflow-deployment.yaml with your S3 Bucket name.
    
```shell
cd numalogic-prometheus/deployment

docker build -t mlflow . && k3d image import docker.io/library/mlflow

kubectl apply -f mlflow-deployment.yaml
```

## Training Workflow Setup
- Create Role and Rolebinding, for argo workflows used for ML training.
```
cd numalogic-prometheus/deployment

kubectl apply -f workflow-role.yaml

kubectl apply -f workflow-rolebinding.yaml
```

- Create argo workflow template. 
   
  Note: Replace `<PROMETHEUS_SERVER>` with your prometheus server endpoint, example: http://prometheus.monitoring.svc.cluster.local:9090
```
kubectl apply -f numalogic-training-workflow-template.yaml
```

## Numalogic Prometheus Pipeline Setup

- Create [Inter-Step Buffer](https://numaflow.numaproj.io/inter-step-buffer/) server.
```shell
kubectl apply -f https://raw.githubusercontent.com/numaproj/numaflow/stable/examples/0-isbsvc-jetstream.yaml
```

- Create numalogic-prometheus pipeline, 
   
    Note: 
    1. Replace `<PROMETHEUS_SERVER>` with your prometheus pushgateway server endpoint, example: http://prometheus-pushgateway.monitoring.svc:9091
    2. Replace `<MLFLOW_S3_ROLE_ARN>` with the AWS role arn created in MLflow setup.
```
kubectl apply -f numalogic-prometheus-pipeline.yaml
```


## Operations

1. To view Numaflow UX:
```
kubectl port-forward svc/numaflow-server 8443  -n numaflow-system
```

Here you can see all the pipelines running on the cluster

2. To view ML flow server:
```
kubectl port-forward svc/mlflow-service 5000 -n numalogic-prometheus
```

Here you can see all the model runs, models saved, etc.

3. To view Prometheus server:
```
kubectl port-forward <prometheus-deployment-xxxxxxxxx-xxxxx> 8490:9090 -n monitoring
```



