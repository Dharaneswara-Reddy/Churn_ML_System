terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Remote state, so losing this working directory does not orphan the
  # infrastructure. Versioning is enabled on the bucket, so a corrupted state
  # can be rolled back.
  # The backend is configured before variables are evaluated, so it cannot
  # reference var.aws_profile and the values are repeated literally here.
  backend "s3" {
    bucket  = "churn-ml-tfstate-183192605828"
    key     = "single-node/terraform.tfstate"
    region  = "us-east-1"
    encrypt = true
    profile = "aiops-deploy"
  }
}

provider "aws" {
  region  = var.region
  profile = var.aws_profile

  default_tags {
    tags = {
      Project     = "churn-ml-system"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}
