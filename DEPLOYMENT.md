# Opinion Market Platform - Deployment Guide

## Overview

The Opinion Market platform is a comprehensive prediction market system with advanced features including real-time trading, AI-powered predictions, enterprise security, and scalable infrastructure. This guide covers the complete deployment process from development to production.

## Architecture

### Core Components

- **API Server**: FastAPI-based REST API with real-time WebSocket support
- **Database**: PostgreSQL for persistent data storage
- **Cache**: Redis for session management and caching
- **Message Queue**: Celery with Redis backend for async tasks
- **Monitoring**: Prometheus + Grafana for observability
- **Security**: Enterprise-grade security with encryption and audit trails
- **AI/ML**: Machine learning models for market predictions

### Infrastructure Stack

- **Container Orchestration**: Kubernetes (EKS)
- **Infrastructure as Code**: Terraform
- **Package Management**: Helm
- **CI/CD**: GitHub Actions
- **Load Balancing**: NGINX Ingress Controller
- **SSL/TLS**: Let's Encrypt with cert-manager
- **Backup**: Automated S3 backups
- **Monitoring**: Prometheus, Grafana, AlertManager

## Prerequisites

### Required Tools

```bash
# Install required tools
brew install kubectl helm terraform awscli docker

# Or on Ubuntu/Debian
sudo apt-get install kubectl helm terraform awscli docker.io
```

### AWS Configuration

1. **AWS CLI Setup**:
   ```bash
   aws configure
   # Enter your AWS Access Key ID, Secret Access Key, and region
   ```

2. **EKS Cluster Access**:
   ```bash
   aws eks update-kubeconfig --region us-east-1 --name opinion-market-production
   ```

3. **Required AWS Services**:
   - EKS (Elastic Kubernetes Service)
   - RDS (PostgreSQL)
   - ElastiCache (Redis)
   - S3 (Backups and file storage)
   - Route53 (DNS)
   - ACM (SSL certificates)
   - CloudWatch (Monitoring)

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/your-org/opinion-market.git
cd opinion-market

# Install Python dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit environment variables
nano .env
```

Required environment variables:
```env
# Database
DATABASE_URL=postgresql://user:password@host:5432/dbname
REDIS_URL=redis://host:6379

# Security
SECRET_KEY=your-secret-key
JWT_SECRET=your-jwt-secret
ENCRYPTION_KEY=your-encryption-key

# Email
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password

# AWS
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=us-east-1
```

### 3. Local Development

```bash
# Start local services
docker-compose up -d

# Run migrations
alembic upgrade head

# Start development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Start Celery worker
celery -A app.celery_app worker --loglevel=info

# Start Celery beat
celery -A app.celery_app beat --loglevel=info
```

### 4. Production Deployment

```bash
# Deploy to production
./deploy.sh production

# Or deploy to staging
./deploy.sh staging
```

## Detailed Deployment Process

### Phase 1: Infrastructure Setup

1. **Terraform Infrastructure**:
   ```bash
   cd deployment/terraform
   terraform init
   terraform plan -var="environment=production"
   terraform apply
   ```

2. **Kubernetes Cluster**:
   - EKS cluster with auto-scaling
   - Multi-AZ deployment
   - Spot instances for cost optimization

3. **Networking**:
   - VPC with public/private subnets
   - NAT gateways for private subnet internet access
   - Security groups for service isolation

### Phase 2: Database and Cache

1. **PostgreSQL RDS**:
   - Multi-AZ deployment
   - Automated backups
   - Performance Insights
   - Read replicas (optional)

2. **Redis ElastiCache**:
   - Cluster mode for high availability
   - Persistence enabled
   - Multi-AZ failover

### Phase 3: Application Deployment

1. **Build and Push Images**:
   ```bash
   docker build -t opinionmarket/api:latest .
   docker push opinionmarket/api:latest
   ```

2. **Helm Deployment**:
   ```bash
   helm upgrade --install opinion-market ./deployment/helm/opinion-market \
     --namespace opinion-market \
     --set environment=production
   ```

3. **Service Configuration**:
   - Load balancer setup
   - SSL certificate configuration
   - DNS routing

### Phase 4: Monitoring and Observability

1. **Prometheus Stack**:
   ```bash
   helm install prometheus prometheus-community/kube-prometheus-stack \
     --namespace monitoring
   ```

2. **Custom Dashboards**:
   - Application metrics
   - Business metrics
   - Infrastructure metrics

3. **Alerting Rules**:
   - Performance alerts
   - Security alerts
   - Business alerts

### Phase 5: Security Configuration

1. **Network Policies**:
   ```bash
   kubectl apply -f deployment/security/network-policies.yaml
   ```

2. **RBAC Configuration**:
   - Service accounts
   - Role bindings
   - Least privilege access

3. **Secrets Management**:
   - AWS Secrets Manager integration
   - Encrypted secrets storage
   - Rotation policies

## Configuration Management

### Helm Values

The platform uses Helm for configuration management. Key configuration files:

- `deployment/helm/opinion-market/values.yaml`: Default values
- `deployment/helm/opinion-market/values-production.yaml`: Production overrides
- `deployment/helm/opinion-market/values-staging.yaml`: Staging overrides

### Environment-Specific Configurations

```yaml
# Production configuration
global:
  environment: production
  domain: opinionmarket.com

app:
  replicas: 3
  resources:
    requests:
      cpu: 500m
      memory: 1Gi
    limits:
      cpu: 1000m
      memory: 2Gi

monitoring:
  enabled: true
  retention: 30d
```

## Monitoring and Alerting

### Key Metrics

1. **Application Metrics**:
   - Request rate and latency
   - Error rates
   - User activity
   - Trading volume

2. **Infrastructure Metrics**:
   - CPU and memory usage
   - Disk space
   - Network traffic
   - Pod health

3. **Business Metrics**:
   - Active markets
   - Trading volume
   - User registrations
   - Revenue metrics

### Alerting Rules

The platform includes comprehensive alerting for:

- **Critical Alerts**: Service outages, security breaches
- **Warning Alerts**: Performance degradation, resource usage
- **Info Alerts**: Business metrics, operational events

### Dashboard Access

- **Grafana**: https://grafana.opinionmarket.com
- **Prometheus**: https://prometheus.opinionmarket.com
- **API Documentation**: https://api.opinionmarket.com/docs

## Backup and Disaster Recovery

### Automated Backups

1. **Database Backups**:
   - Daily automated backups
   - Point-in-time recovery
   - Cross-region replication

2. **Application Data**:
   - S3 bucket versioning
   - Lifecycle policies
   - Cross-region replication

3. **Configuration**:
   - Git-based configuration
   - Terraform state backups
   - Helm releases tracking

### Recovery Procedures

1. **Database Recovery**:
   ```bash
   # Restore from snapshot
   aws rds restore-db-instance-from-db-snapshot \
     --db-instance-identifier opinion-market-restored \
     --db-snapshot-identifier snapshot-id
   ```

2. **Application Recovery**:
   ```bash
   # Redeploy application
   helm upgrade opinion-market ./deployment/helm/opinion-market
   ```

## Security Considerations

### Data Protection

1. **Encryption**:
   - Data at rest (AES-256)
   - Data in transit (TLS 1.3)
   - Database encryption
   - S3 encryption

2. **Access Control**:
   - IAM roles and policies
   - Kubernetes RBAC
   - Network policies
   - API authentication

3. **Compliance**:
   - GDPR compliance
   - SOC 2 Type II
   - PCI DSS (if applicable)
   - Regular security audits

### Security Monitoring

1. **Threat Detection**:
   - AWS GuardDuty
   - CloudWatch security events
   - Custom security metrics

2. **Vulnerability Management**:
   - Regular dependency updates
   - Container scanning
   - Security patches

## Performance Optimization

### Scaling Strategies

1. **Horizontal Pod Autoscaling**:
   ```yaml
   autoscaling:
     enabled: true
     minReplicas: 3
     maxReplicas: 10
     targetCPUUtilizationPercentage: 70
   ```

2. **Database Optimization**:
   - Connection pooling
   - Query optimization
   - Read replicas
   - Caching strategies

3. **CDN Configuration**:
   - CloudFront distribution
   - Edge caching
   - Geographic distribution

### Performance Testing

```bash
# Run performance tests
kubectl apply -f deployment/testing/performance-tests.yaml

# Monitor results
kubectl logs job/performance-test -n opinion-market
```

## Troubleshooting

### Common Issues

1. **Pod Startup Issues**:
   ```bash
   kubectl describe pod <pod-name> -n opinion-market
   kubectl logs <pod-name> -n opinion-market
   ```

2. **Database Connection Issues**:
   ```bash
   kubectl exec -it deployment/opinion-market-api -n opinion-market -- pg_isready
   ```

3. **Redis Connection Issues**:
   ```bash
   kubectl exec -it deployment/opinion-market-api -n opinion-market -- redis-cli ping
   ```

### Debugging Commands

```bash
# Check cluster status
kubectl get nodes
kubectl get pods -n opinion-market

# Check services
kubectl get svc -n opinion-market
kubectl get ingress -n opinion-market

# Check logs
kubectl logs -f deployment/opinion-market-api -n opinion-market

# Check events
kubectl get events -n opinion-market --sort-by='.lastTimestamp'
```

## Maintenance

### Regular Maintenance Tasks

1. **Security Updates**:
   - Monthly dependency updates
   - Security patch deployment
   - Certificate renewal

2. **Performance Optimization**:
   - Database maintenance
   - Cache optimization
   - Resource scaling

3. **Backup Verification**:
   - Weekly backup tests
   - Recovery procedure validation
   - Data integrity checks

### Update Procedures

1. **Application Updates**:
   ```bash
   # Update application
   helm upgrade opinion-market ./deployment/helm/opinion-market
   ```

2. **Infrastructure Updates**:
   ```bash
   # Update infrastructure
   terraform plan
   terraform apply
   ```

3. **Database Updates**:
   ```bash
   # Run migrations
   kubectl exec -it deployment/opinion-market-api -n opinion-market -- alembic upgrade head
   ```

## Support and Documentation

### Additional Resources

- [API Documentation](https://api.opinionmarket.com/docs)
- [Architecture Diagrams](docs/architecture.md)
- [Development Guide](docs/development.md)
- [Contributing Guidelines](CONTRIBUTING.md)

### Contact Information

- **Technical Support**: tech-support@opinionmarket.com
- **Security Issues**: security@opinionmarket.com
- **Documentation**: docs@opinionmarket.com

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
