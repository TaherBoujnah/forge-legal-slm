provider "aws" {
  region = "eu-central-1"
}

# Generate a random string for unique bucket names
resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# ==================================================
# PHASE 1: ECR REPOSITORY (Docker Image)
# ==================================================
resource "aws_ecr_repository" "lambda_inference_repo" {
  name                 = "forge-legal-slm-inference"
  image_tag_mutability = "MUTABLE"
  force_delete         = true
}

# ==================================================
# PHASE 2: S3 BUCKET (Model Artifacts / GGUF)
# ==================================================
resource "aws_s3_bucket" "model_artifacts" {
  bucket        = "forge-legal-slm-artifacts-${random_id.bucket_suffix.hex}"
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "model_artifacts_privacy" {
  bucket                  = aws_s3_bucket.model_artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

output "ecr_repository_url" {
  description = "The URL to push your Docker container to"
  value       = aws_ecr_repository.lambda_inference_repo.repository_url
}

# ==================================================
# PHASE 3 & 4: API GATEWAY, LAMBDA, & S3 FRONTEND
# ==================================================

# 1. IAM Role for Lambda
resource "aws_iam_role" "lambda_exec" {
  name = "forge-lambda-exec-role-${random_id.bucket_suffix.hex}"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

# 2. Give Lambda permission to read from the Model S3 Bucket and write logs
resource "aws_iam_role_policy" "lambda_s3_policy" {
  name = "forge-lambda-s3-policy"
  role = aws_iam_role.lambda_exec.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = ["s3:GetObject", "s3:ListBucket"]
        Effect = "Allow"
        Resource = [
          aws_s3_bucket.model_artifacts.arn,
          "${aws_s3_bucket.model_artifacts.arn}/*"
        ]
      },
      {
        Action = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        Effect = "Allow"
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

# 3. The Lambda Function itself (Using the Docker Image)
resource "aws_lambda_function" "inference" {
  function_name = "forge-legal-slm-inference"
  role          = aws_iam_role.lambda_exec.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.lambda_inference_repo.repository_url}:latest"
  
  # Adjusted to meet AWS default account quota limits
  memory_size   = 3008 
  timeout       = 300  

  # Give Lambda a 4GB hard drive so the AI model fits!
  ephemeral_storage {
    size = 4096
  }

  environment {
    variables = {
      MODEL_BUCKET = aws_s3_bucket.model_artifacts.bucket
    }
  }
}

# 4. API Gateway (HTTP API) to expose the Lambda to the public web
resource "aws_apigatewayv2_api" "api" {
  name          = "forge-legal-slm-api"
  protocol_type = "HTTP"
  target        = aws_lambda_function.inference.arn
  
  cors_configuration {
    allow_origins = ["*"]
    allow_methods = ["POST", "OPTIONS"]
    allow_headers = ["content-type"]
  }
}

# Allow API Gateway to trigger the Lambda function
resource "aws_lambda_permission" "apigw" {
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.inference.arn
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.api.execution_arn}/*/*"
}

# 5. S3 Bucket for Static Website Hosting (The Frontend)
resource "aws_s3_bucket" "frontend" {
  bucket        = "forge-legal-slm-frontend-${random_id.bucket_suffix.hex}"
  force_destroy = true
}

resource "aws_s3_bucket_website_configuration" "frontend" {
  bucket = aws_s3_bucket.frontend.id
  index_document { suffix = "index.html" }
}

resource "aws_s3_bucket_public_access_block" "frontend_public_access" {
  bucket = aws_s3_bucket.frontend.id
  block_public_acls       = false
  block_public_policy     = false
  ignore_public_acls      = false
  restrict_public_buckets = false
}

resource "aws_s3_bucket_policy" "frontend_policy" {
  bucket = aws_s3_bucket.frontend.id
  depends_on = [aws_s3_bucket_public_access_block.frontend_public_access]
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "s3:GetObject"
      Effect    = "Allow"
      Resource  = "${aws_s3_bucket.frontend.arn}/*"
      Principal = "*"
    }]
  })
}

output "api_gateway_url" {
  description = "The public API endpoint for your Lambda function"
  value       = aws_apigatewayv2_api.api.api_endpoint
}

output "frontend_bucket_name" {
  description = "The bucket where you will upload index.html"
  value       = aws_s3_bucket.frontend.bucket
}

output "frontend_website_url" {
  description = "The public URL for your deployed dashboard!"
  value       = aws_s3_bucket_website_configuration.frontend.website_endpoint
}