terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }
}

provider "aws" {
  region = "eu-central-1" # Frankfurt region
  default_tags {
    tags = {
      Project     = "Forge-Legal-SLM"
      Environment = "Production"
      ManagedBy   = "Terraform"
    }
  }
}

# S3 bucket names must be globally unique across all of AWS.
resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# Provisioning S3 Bucket for Model Artifacts
resource "aws_s3_bucket" "model_artifacts" {
  bucket = "forge-legal-slm-artifacts-${random_id.bucket_suffix.hex}"
}

# Enable versioning so every time you upload a new model, the old one is saved
resource "aws_s3_bucket_versioning" "model_artifacts_versioning" {
  bucket = aws_s3_bucket.model_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Ensure the bucket is private (Security Best Practice)
resource "aws_s3_bucket_public_access_block" "model_artifacts_privacy" {
  bucket                  = aws_s3_bucket.model_artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Provisioning Elastic Container Registry (ECR)
# This will hold the Docker image for our AWS Lambda function
resource "aws_ecr_repository" "lambda_inference_repo" {
  name                 = "forge-legal-slm-inference"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

# Defining outputs for the terminal
output "s3_bucket_name" {
  description = "The name of the bucket to upload your model weights to"
  value       = aws_s3_bucket.model_artifacts.bucket
}

output "ecr_repository_url" {
  description = "The URL to push your Docker container to"
  value       = aws_ecr_repository.lambda_inference_repo.repository_url
}