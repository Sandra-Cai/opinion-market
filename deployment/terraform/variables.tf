variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
  
  validation {
    condition     = contains(["development", "staging", "production"], var.environment)
    error_message = "Environment must be one of: development, staging, production."
  }
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "opinion-market"
}

variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = "opinionmarket.com"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "Availability zones"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
}

variable "rds_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.micro"
}

variable "rds_allocated_storage" {
  description = "RDS allocated storage in GB"
  type        = number
  default     = 20
}

variable "rds_max_allocated_storage" {
  description = "RDS maximum allocated storage in GB"
  type        = number
  default     = 100
}

variable "redis_node_type" {
  description = "ElastiCache Redis node type"
  type        = string
  default     = "cache.t3.micro"
}

variable "enable_spot_instances" {
  description = "Enable spot instances for cost optimization"
  type        = bool
  default     = false
}

variable "spot_instance_types" {
  description = "Instance types for spot instances"
  type        = list(string)
  default     = ["t3.medium", "t3.small"]
}

variable "on_demand_instance_types" {
  description = "Instance types for on-demand instances"
  type        = list(string)
  default     = ["t3.medium"]
}

variable "min_nodes" {
  description = "Minimum number of nodes"
  type        = number
  default     = 1
}

variable "max_nodes" {
  description = "Maximum number of nodes"
  type        = number
  default     = 5
}

variable "desired_nodes" {
  description = "Desired number of nodes"
  type        = number
  default     = 2
}

variable "enable_autoscaling" {
  description = "Enable cluster autoscaling"
  type        = bool
  default     = true
}

variable "enable_monitoring" {
  description = "Enable CloudWatch monitoring"
  type        = bool
  default     = true
}

variable "enable_logging" {
  description = "Enable CloudWatch logging"
  type        = bool
  default     = true
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 30
}

variable "enable_backup" {
  description = "Enable automated backups"
  type        = bool
  default     = true
}

variable "backup_retention_days" {
  description = "Backup retention period in days"
  type        = number
  default     = 30
}

variable "enable_encryption" {
  description = "Enable encryption at rest"
  type        = bool
  default     = true
}

variable "enable_ssl" {
  description = "Enable SSL/TLS encryption"
  type        = bool
  default     = true
}

variable "enable_waf" {
  description = "Enable AWS WAF"
  type        = bool
  default     = true
}

variable "enable_shield" {
  description = "Enable AWS Shield Advanced"
  type        = bool
  default     = false
}

variable "enable_guardduty" {
  description = "Enable AWS GuardDuty"
  type        = bool
  default     = true
}

variable "enable_config" {
  description = "Enable AWS Config"
  type        = bool
  default     = true
}

variable "enable_cloudtrail" {
  description = "Enable AWS CloudTrail"
  type        = bool
  default     = true
}

variable "enable_vpc_flow_logs" {
  description = "Enable VPC Flow Logs"
  type        = bool
  default     = true
}

variable "enable_nat_gateway" {
  description = "Enable NAT Gateway"
  type        = bool
  default     = true
}

variable "single_nat_gateway" {
  description = "Use single NAT Gateway for cost optimization"
  type        = bool
  default     = false
}

variable "enable_vpn" {
  description = "Enable VPN access"
  type        = bool
  default     = false
}

variable "vpn_cidr" {
  description = "CIDR block for VPN access"
  type        = string
  default     = "10.1.0.0/16"
}

variable "enable_bastion" {
  description = "Enable bastion host"
  type        = bool
  default     = false
}

variable "bastion_instance_type" {
  description = "Bastion host instance type"
  type        = string
  default     = "t3.micro"
}

variable "enable_elasticsearch" {
  description = "Enable Elasticsearch for logging"
  type        = bool
  default     = false
}

variable "elasticsearch_instance_type" {
  description = "Elasticsearch instance type"
  type        = string
  default     = "t3.small.elasticsearch"
}

variable "enable_redis_cluster" {
  description = "Enable Redis cluster mode"
  type        = bool
  default     = false
}

variable "redis_num_cache_nodes" {
  description = "Number of Redis cache nodes"
  type        = number
  default     = 1
}

variable "enable_rds_multi_az" {
  description = "Enable RDS Multi-AZ deployment"
  type        = bool
  default     = true
}

variable "enable_rds_read_replica" {
  description = "Enable RDS read replica"
  type        = bool
  default     = false
}

variable "enable_rds_performance_insights" {
  description = "Enable RDS Performance Insights"
  type        = bool
  default     = true
}

variable "rds_performance_insights_retention" {
  description = "RDS Performance Insights retention period"
  type        = number
  default     = 7
}

variable "enable_s3_versioning" {
  description = "Enable S3 versioning"
  type        = bool
  default     = true
}

variable "enable_s3_lifecycle" {
  description = "Enable S3 lifecycle policies"
  type        = bool
  default     = true
}

variable "s3_lifecycle_days" {
  description = "S3 lifecycle transition days"
  type        = number
  default     = 90
}

variable "enable_cloudfront" {
  description = "Enable CloudFront CDN"
  type        = bool
  default     = true
}

variable "enable_route53_health_checks" {
  description = "Enable Route53 health checks"
  type        = bool
  default     = true
}

variable "enable_route53_failover" {
  description = "Enable Route53 failover routing"
  type        = bool
  default     = false
}

variable "enable_ses" {
  description = "Enable Amazon SES for email"
  type        = bool
  default     = true
}

variable "enable_sns" {
  description = "Enable Amazon SNS for notifications"
  type        = bool
  default     = true
}

variable "enable_sqs" {
  description = "Enable Amazon SQS for queuing"
  type        = bool
  default     = false
}

variable "enable_lambda" {
  description = "Enable AWS Lambda functions"
  type        = bool
  default     = false
}

variable "enable_api_gateway" {
  description = "Enable API Gateway"
  type        = bool
  default     = false
}

variable "enable_cognito" {
  description = "Enable Amazon Cognito for authentication"
  type        = bool
  default     = false
}

variable "enable_kms" {
  description = "Enable AWS KMS for encryption"
  type        = bool
  default     = true
}

variable "enable_secrets_manager" {
  description = "Enable AWS Secrets Manager"
  type        = bool
  default     = true
}

variable "enable_parameter_store" {
  description = "Enable AWS Systems Manager Parameter Store"
  type        = bool
  default     = true
}

variable "enable_cloudwatch_alarms" {
  description = "Enable CloudWatch alarms"
  type        = bool
  default     = true
}

variable "cpu_threshold" {
  description = "CPU utilization threshold for alarms"
  type        = number
  default     = 80
}

variable "memory_threshold" {
  description = "Memory utilization threshold for alarms"
  type        = number
  default     = 80
}

variable "disk_threshold" {
  description = "Disk utilization threshold for alarms"
  type        = number
  default     = 85
}

variable "enable_cost_optimization" {
  description = "Enable cost optimization features"
  type        = bool
  default     = true
}

variable "enable_resource_tagging" {
  description = "Enable automatic resource tagging"
  type        = bool
  default     = true
}

variable "tags" {
  description = "Additional tags for resources"
  type        = map(string)
  default     = {}
}
