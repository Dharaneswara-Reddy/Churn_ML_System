variable "region" {
  description = "AWS region."
  type        = string
  default     = "us-east-1"
}

variable "aws_profile" {
  description = "Local AWS CLI profile used to apply this configuration."
  type        = string
  default     = "aiops-deploy"
}

variable "environment" {
  description = "Environment name, used in resource names and tags."
  type        = string
  default     = "prod"
}

variable "instance_type" {
  description = <<-EOT
    EC2 instance type.

    t3.small is sized from measurement rather than habit: the API process peaks
    at 183 MiB with the model loaded, and PostgreSQL, the outbox worker and the
    container runtime fit alongside it inside 2 GiB. Move to t3.medium if the
    memory alarm fires.
  EOT
  type        = string
  default     = "t3.small"
}

variable "root_volume_gb" {
  description = "Root EBS volume size. Holds the OS, images, model bundle and the PostgreSQL data directory."
  type        = number
  default     = 30
}

variable "artifacts_bucket" {
  description = "S3 bucket holding the signed model bundle."
  type        = string
  default     = "churn-ml-artifacts-183192605828"
}

variable "repo_url" {
  description = "Public git repository cloned onto the instance at boot."
  type        = string
  default     = "https://github.com/GojoV339/Churn_ML_System.git"
}

variable "repo_ref" {
  description = "Git ref to deploy. Pin to a tag or commit for a reproducible rollout."
  type        = string
  default     = "main"
}

variable "allowed_http_cidrs" {
  description = <<-EOT
    CIDRs permitted to reach the API on port 80.

    Defaults to the whole internet because the service is a public demo API
    guarded by an API key. Narrow this to your own address if you would rather
    it not be reachable from everywhere.
  EOT
  type        = list(string)
  default     = ["0.0.0.0/0"]
}
