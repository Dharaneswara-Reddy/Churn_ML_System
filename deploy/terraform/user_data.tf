# Instance bootstrap.
#
# The image tag is pinned to a git commit rather than "latest": a replacement
# instance must come up running the same code as its predecessor, and "latest"
# silently breaks that the moment anyone pushes.
variable "image_tag" {
  description = "ECR image tag to deploy. Pinned to a git commit for reproducibility."
  type        = string
  default     = "0f6ae6c"
}

locals {
  ecr_registry = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.region}.amazonaws.com"

  user_data = templatefile("${path.module}/user_data.sh.tftpl", {
    region           = var.region
    environment      = var.environment
    artifacts_bucket = var.artifacts_bucket
    repo_url         = var.repo_url
    repo_ref         = var.repo_ref
    aws_profile      = var.aws_profile
    ecr_registry     = local.ecr_registry
    api_image        = "${local.ecr_registry}/churn-ml-api:${var.image_tag}"
    training_image   = "${local.ecr_registry}/churn-ml-training:${var.image_tag}"
    log_group        = aws_cloudwatch_log_group.app.name
  })
}
