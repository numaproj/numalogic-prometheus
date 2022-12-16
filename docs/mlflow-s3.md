# MLflow with S3:

## AWS S3 Bucket Setup:
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

## MLflow Server Setup:
 - Replace [MLFLOW_S3_USER_ARN](https://github.com/numaproj/numalogic-prometheus/blob/main/manifests/mlflow/mlflow-deployment.yaml#L14) in mlflow-deployment.yaml with your AWS User arn. 
 - Replace [MLFLOW_BUCKET](https://github.com/numaproj/numalogic-prometheus/blob/main/manifests/mlflow/mlflow-deployment.yaml#L28) in mlflow-deployment.yaml with your S3 Bucket name.
 - Replace mlflow-deployment-local.yaml with mlflow-deployment.yaml in [kustomization.yaml](https://github.com/numaproj/numalogic-prometheus/blob/main/manifests/mlflow/kustomization.yaml)
 

## Training Workflow Setup
 - Replace the `<MLFLOW_S3_ROLE_ARN>` under annotations in [numalogic-training-workflow-template.yaml]((https://github.com/numaproj/numalogic-prometheus/blob/main/manifests/argoworkflows/numalogic-training-workflow-template.yaml))


Once you made all the above changes, run the below command:
```shell
  kustomize build numalogic-prometheus/manifests/ | kubectl apply -f - 
```