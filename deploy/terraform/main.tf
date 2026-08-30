# Single-node deployment of the churn ML system.
#
# One EC2 instance runs the API, the outbox worker and PostgreSQL under docker
# compose. This is the cheapest shape that still exercises the real serving path,
# and it is honest about what it is: a single node, with no rolling deploys and no
# replica redundancy. The multi-replica work (PostgreSQL advisory-lock leader
# election, shared bundle volume, HPA) lives in deploy/kubernetes/ for when this
# outgrows one box.
#
# Deliberate choices:
#
#   * No SSH key pair and no inbound port 22. Access is via SSM Session Manager,
#     which authenticates with IAM, is audited in CloudTrail, and removes the
#     standing risk of an internet-reachable sshd with a key nobody rotates.
#   * The instance profile can read exactly two things: the model bundle prefix in
#     S3, and the /churn-ml/<env>/ SSM parameters. Not the whole bucket, not all
#     parameters.
#   * IMDSv2 is required, so a request-forgery bug in the app cannot be used to
#     read instance credentials through the v1 metadata endpoint.

data "aws_vpc" "default" {
  default = true
}

# Not every availability zone offers every instance type — us-east-1e has no
# t3.small at all, and the default VPC has a subnet there. Picking "the first
# subnet" therefore fails on an AZ-dependent coin flip rather than anything to do
# with this configuration, so the AZ list is derived from what the chosen instance
# type is actually offered in.
data "aws_ec2_instance_type_offerings" "supported" {
  filter {
    name   = "instance-type"
    values = [var.instance_type]
  }

  location_type = "availability-zone"
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }

  filter {
    name   = "availability-zone"
    values = data.aws_ec2_instance_type_offerings.supported.locations
  }
}

# Canonical's Ubuntu 24.04 LTS, resolved at apply time rather than hardcoded so
# the deployment does not pin a stale, unpatched image.
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

data "aws_caller_identity" "current" {}

locals {
  name = "churn-ml-${var.environment}"
}

# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

resource "aws_security_group" "app" {
  name        = "${local.name}-sg"
  description = "Churn ML API. HTTP in, everything out."
  vpc_id      = data.aws_vpc.default.id

  # No ingress on 22: access is via SSM Session Manager.
  ingress {
    description = "API traffic"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = var.allowed_http_cidrs
  }

  # Egress is restricted to the two ports the node genuinely needs: HTTPS for
  # apt, ECR, S3 and the SSM agent, and HTTP for a handful of apt mirrors and
  # redirects. It is not narrowed by destination because those are AWS service
  # ranges plus the Ubuntu and Docker mirrors, which change; pinning them would
  # break the instance on a routine upstream change rather than protect it.
  #
  # Restricting the *ports* is still worth doing on its own: it stops an
  # arbitrary outbound channel on some other port, which is what a compromised
  # process would reach for.
  #
  # Removing internet egress entirely means VPC endpoints for S3, ECR, SSM and
  # CloudWatch. The S3 gateway endpoint is free; the interface endpoints are
  # roughly $7/month each, which is a large fraction of this deployment's total
  # cost. See .trivyignore.
  egress {
    description = "HTTPS for ECR, S3, SSM and package repositories"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description = "HTTP for apt mirrors and redirects"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description = "DNS"
    from_port   = 53
    to_port     = 53
    protocol    = "udp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${local.name}-sg" }
}

# ---------------------------------------------------------------------------
# Instance identity
# ---------------------------------------------------------------------------

resource "aws_iam_role" "instance" {
  name = "${local.name}-instance-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

# Session Manager access, so the instance needs no inbound SSH.
resource "aws_iam_role_policy_attachment" "ssm_core" {
  role       = aws_iam_role.instance.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy" "app" {
  name = "${local.name}-app"
  role = aws_iam_role.instance.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        # Exactly the model prefix — not the bucket.
        Sid      = "ReadModelBundle"
        Effect   = "Allow"
        Action   = ["s3:GetObject"]
        Resource = ["arn:aws:s3:::${var.artifacts_bucket}/model/*"]
      },
      {
        Sid      = "ListModelPrefix"
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = ["arn:aws:s3:::${var.artifacts_bucket}"]
        Condition = {
          StringLike = { "s3:prefix" = ["model/*"] }
        }
      },
      {
        # Exactly this environment's parameters.
        Sid    = "ReadOwnSecrets"
        Effect = "Allow"
        Action = ["ssm:GetParameter", "ssm:GetParameters", "ssm:GetParametersByPath"]
        Resource = [
          "arn:aws:ssm:${var.region}:${data.aws_caller_identity.current.account_id}:parameter/churn-ml/${var.environment}",
          "arn:aws:ssm:${var.region}:${data.aws_caller_identity.current.account_id}:parameter/churn-ml/${var.environment}/*",
        ]
      },
      {
        Sid      = "DecryptOwnSecrets"
        Effect   = "Allow"
        Action   = ["kms:Decrypt"]
        Resource = ["arn:aws:kms:${var.region}:${data.aws_caller_identity.current.account_id}:alias/aws/ssm"]
      },
      {
        # GetAuthorizationToken is account-scoped by design: the ECR API grants
        # the login token globally and does not accept a repository resource.
        Sid      = "EcrAuth"
        Effect   = "Allow"
        Action   = ["ecr:GetAuthorizationToken"]
        Resource = ["*"]
      },
      {
        # Pulling layers IS scoped, to exactly this project's two repositories.
        Sid    = "EcrPull"
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:BatchGetImage",
          "ecr:GetDownloadUrlForLayer",
        ]
        Resource = [
          "arn:aws:ecr:${var.region}:${data.aws_caller_identity.current.account_id}:repository/churn-ml-api",
          "arn:aws:ecr:${var.region}:${data.aws_caller_identity.current.account_id}:repository/churn-ml-training",
        ]
      },
      {
        Sid      = "PublishLogs"
        Effect   = "Allow"
        Action   = ["logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"]
        Resource = ["${aws_cloudwatch_log_group.app.arn}:*"]
      },
    ]
  })
}

resource "aws_iam_instance_profile" "instance" {
  name = "${local.name}-instance-profile"
  role = aws_iam_role.instance.name
}

resource "aws_cloudwatch_log_group" "app" {
  name              = "/churn-ml/${var.environment}"
  retention_in_days = 14
}

# ---------------------------------------------------------------------------
# Instance
# ---------------------------------------------------------------------------

resource "aws_instance" "app" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  subnet_id              = data.aws_subnets.default.ids[0]
  vpc_security_group_ids = [aws_security_group.app.id]
  iam_instance_profile   = aws_iam_instance_profile.instance.name

  user_data                   = local.user_data
  user_data_replace_on_change = true

  metadata_options {
    http_endpoint = "enabled"
    # IMDSv2 only. A server-side request forgery in the application cannot then
    # read instance credentials from the unauthenticated v1 endpoint.
    http_tokens                 = "required"
    http_put_response_hop_limit = 2 # containers need one extra hop
    instance_metadata_tags      = "disabled"
  }

  root_block_device {
    volume_size           = var.root_volume_gb
    volume_type           = "gp3"
    encrypted             = true
    delete_on_termination = true
  }

  tags = { Name = local.name }
}

# A stable address that survives stop/start and instance replacement.
resource "aws_eip" "app" {
  instance = aws_instance.app.id
  domain   = "vpc"
  tags     = { Name = "${local.name}-eip" }
}

# ---------------------------------------------------------------------------
# Alarms
# ---------------------------------------------------------------------------

resource "aws_cloudwatch_metric_alarm" "cpu" {
  alarm_name          = "${local.name}-cpu-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = 300
  statistic           = "Average"
  threshold           = 85
  alarm_description   = "Sustained CPU pressure. Measured capacity is ~32 rps per uvicorn worker; this is the signal to scale up or out."
  dimensions          = { InstanceId = aws_instance.app.id }
}

resource "aws_cloudwatch_metric_alarm" "status" {
  alarm_name          = "${local.name}-status-check-failed"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "StatusCheckFailed"
  namespace           = "AWS/EC2"
  period              = 60
  statistic           = "Maximum"
  threshold           = 0
  alarm_description   = "Instance or system status check failing."
  dimensions          = { InstanceId = aws_instance.app.id }
}
