# numalogic-prometheus

## Prerequisites
- [Numaflow](https://numaflow.numaproj.io/quick-start/#installation)
- [Prometheus](https://prometheus.io/docs/prometheus/latest/getting_started/)


## Instructions
1. Fork and clone the repository to your local and build numalogic-prometheus docker image.

```shell
git@github.com:numaproj/numalogic-prometheus.git

docker build -t numalogic-prometheus . && k3d image import docker.io/library/numalogic-prometheus
```

2. Create numalogic-prometheus namespace.
```shell
kubectl create namespace numalogic-prometheus

kubectl config set-context --current --namespace=numalogic-prometheus
```

3. MLflow S3 bucket setup
- Create an [AWS S3 bucket](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html), called `mlflow-nuamlogic-prometheus`
- Create an [AWS IAM Role](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_create.html), with the following policy.
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
                "arn:aws:s3:::mlflow-nuamlogic-prometheus",
                "arn:aws:s3:::mlflow-nuamlogic-prometheus/*",
            ]
        }
    ]
}
```
- Attach the created role to the s3 bucket under permissions.
```shell
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AddCannedAcl",
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::<account>:role/<role_name>"
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
                "arn:aws:s3:::mlflow-nuamlogic-prometheus",
                "arn:aws:s3:::mlflow-nuamlogic-prometheus/*"
            ]
        }
    ]
}
```

- Create an AWS User for the role created, by adding the role under permissions.

4. Build Mlflow docker image and install MLflow server. 

    Note: Replace `<MLFLOW_S3_USER_ARN>` and `<MLFLOW_BUCKET>` in mlflow-deployment.yaml with the user arn and bucket name created in above step.

```shell
cd numalogic-prometheus/deployment

docker build -t mlflow .

kubectl apply -f mlflow-deployment.yaml
```

4. Install redis in numalogic-prometheus namespace and copy the password.
```shell
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install numalogic bitnami/redis-cluster
echo $(kubectl get secret --namespace "numalogic-prometheus" numalogic-redis-cluster -o jsonpath="{.data.redis-password}" | base64 -d)
```

4. Create [Inter-Step Buffer](https://numaflow.numaproj.io/inter-step-buffer/) server.
```shell
kubectl apply -f https://raw.githubusercontent.com/numaproj/numaflow/stable/examples/0-isbsvc-jetstream.yaml
```

5. Create workflow role, for argo workflows.

```
kubectl apply -f workflow-role.yaml

kubectl apply -f workflow-rolebinding.yaml
```

6. Create argo workflow template. 
   
    Note:
     1. Replace `<PROMETHEUS_SERVER>` with your prometheus server endpoint, example: http://prometheus.monitoring.svc.cluster.local:9090
     2. Replace `<NUMALOGIC_PROMETHEUS_IMAGE>` with the numalogic-prometheus image created in step 1.
```
kubectl apply -f numalogic-training-workflow-template.yaml
```

7. Create numalogic-prometheus pipeline, 
   
    Note: 
    1. Replace `<NUMALOGIC_PROMETHEUS_IMAGE>` with the numalogic-prometheus image created in step 1.
    2. Replace `<PROMETHEUS_SERVER>` with your prometheus pushgateway server endpoint, example: http://prometheus-pushgateway.monitoring.svc:9091
    3. Replace `<MLFLOW_S3_ROLE_ARN>` with the role arn created in step 3.
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



