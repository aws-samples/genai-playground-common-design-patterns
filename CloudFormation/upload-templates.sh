#!/bin/bash

set -x

S3_BUCKET=${1}
S3_PREFIX=${2}

aws s3 cp root-template.yaml s3://$S3_BUCKET/$S3_PREFIX/root-template.yaml
aws s3 cp vpc-template.yaml s3://$S3_BUCKET/$S3_PREFIX/vpc-template.yaml
aws s3 cp image-template.yaml s3://$S3_BUCKET/$S3_PREFIX/image-template.yaml
aws s3 cp sagemaker-template.yaml s3://$S3_BUCKET/$S3_PREFIX/sagemaker-template.yaml
aws s3 cp fastchat-template.yaml s3://$S3_BUCKET/$S3_PREFIX/fastchat-template.yaml