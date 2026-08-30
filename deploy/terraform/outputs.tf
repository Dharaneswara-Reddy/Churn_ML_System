output "instance_id" {
  description = "EC2 instance id. Use it to open a shell: aws ssm start-session --target <id>"
  value       = aws_instance.app.id
}

output "public_ip" {
  description = "Stable Elastic IP of the API."
  value       = aws_eip.app.public_ip
}

output "api_url" {
  description = "Base URL of the deployed API."
  value       = "http://${aws_eip.app.public_ip}"
}

output "health_url" {
  value = "http://${aws_eip.app.public_ip}/health"
}

output "ready_url" {
  description = "Readiness probe; runs a real one-row prediction through the serving path."
  value       = "http://${aws_eip.app.public_ip}/ready"
}

output "log_group" {
  description = "CloudWatch log group receiving api and worker container logs."
  value       = aws_cloudwatch_log_group.app.name
}

output "session_command" {
  description = "Open a shell on the instance (no SSH port is exposed)."
  value       = "aws ssm start-session --target ${aws_instance.app.id} --profile ${var.aws_profile} --region ${var.region}"
}

output "api_key_command" {
  description = "Retrieve the API key clients must send as X-API-Key."
  value       = "aws ssm get-parameter --name /churn-ml/${var.environment}/api_key --with-decryption --query Parameter.Value --output text --profile ${var.aws_profile} --region ${var.region}"
}
