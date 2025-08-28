# CI/CD Pipeline Setup and Troubleshooting

## Overview

The Opinion Market platform uses GitHub Actions for continuous integration and deployment. The CI/CD pipeline is designed to be robust and handle failures gracefully.

## Pipeline Structure

### 1. Test Workflow (`test.yml`)
- **Triggers**: Push to main/develop, Pull requests
- **Purpose**: Run tests and code quality checks
- **Services**: PostgreSQL, Redis
- **Steps**:
  - Install dependencies
  - Run linting (flake8, black, isort)
  - Run unit tests with coverage
  - Run simple tests
  - Upload coverage to Codecov

### 2. Build Workflow (`build.yml`)
- **Triggers**: Push to main/develop, Pull requests
- **Purpose**: Build and push Docker images
- **Steps**:
  - Set up Docker Buildx
  - Login to GitHub Container Registry
  - Build multi-platform images (amd64, arm64)
  - Push with appropriate tags

### 3. Deploy Workflow (`deploy.yml`)
- **Triggers**: Push to develop (staging), Release (production)
- **Purpose**: Deploy to staging and production environments
- **Steps**:
  - Deploy using Helm
  - Run health checks
  - Send notifications

### 4. Full CI/CD Workflow (`ci-cd.yml`)
- **Triggers**: Push to main/develop, Pull requests, Release
- **Purpose**: Complete pipeline with all features
- **Includes**:
  - Security scanning (Trivy, Bandit, Safety)
  - Code quality and testing
  - Docker build and push
  - Deployment to staging/production
  - Performance testing
  - Security compliance checks
  - Documentation generation

## Common Issues and Solutions

### 1. Test Failures

**Issue**: Tests failing due to missing dependencies
```bash
# Solution: Ensure all required packages are in requirements.txt
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

**Issue**: Database connection failures
```bash
# Solution: Check PostgreSQL service is running
# The workflow includes PostgreSQL service with health checks
```

**Issue**: Redis connection failures
```bash
# Solution: Check Redis service is running
# The workflow includes Redis service with health checks
```

### 2. Linting Failures

**Issue**: Code formatting issues
```bash
# Solution: Run black to format code
black app/ tests/

# Solution: Run isort to sort imports
isort app/ tests/
```

**Issue**: Flake8 violations
```bash
# Solution: Fix code style issues
# Check .flake8 configuration for rules
```

### 3. Docker Build Failures

**Issue**: Build context too large
```bash
# Solution: Check .dockerignore file
# Exclude unnecessary files from build context
```

**Issue**: Multi-platform build failures
```bash
# Solution: Ensure Dockerfile is compatible with both amd64 and arm64
# Use platform-agnostic base images
```

### 4. Deployment Failures

**Issue**: Helm deployment timeout
```bash
# Solution: Check Kubernetes cluster resources
# Increase timeout in deployment command
```

**Issue**: Missing secrets
```bash
# Solution: Ensure all required secrets are set in GitHub repository
# Required secrets:
# - STAGING_KUBECONFIG
# - PRODUCTION_KUBECONFIG
# - STAGING_DATABASE_URL
# - PRODUCTION_DATABASE_URL
# - STAGING_REDIS_URL
# - PRODUCTION_REDIS_URL
# - STAGING_DATABASE_PASSWORD
# - PRODUCTION_DATABASE_PASSWORD
# - STAGING_SMTP_PASSWORD
# - PRODUCTION_SMTP_PASSWORD
# - STAGING_JWT_SECRET
# - PRODUCTION_JWT_SECRET
# - STAGING_ENCRYPTION_KEY
# - PRODUCTION_ENCRYPTION_KEY
# - SLACK_WEBHOOK_URL
```

### 5. Health Check Failures

**Issue**: Application not responding to health checks
```bash
# Solution: Ensure health endpoints are implemented
# - /health - Basic health check
# - /ready - Readiness check
# - /metrics - Prometheus metrics
```

## Local Development Setup

### 1. Install Dependencies
```bash
# Install Python dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### 2. Run Tests Locally
```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/api/ -v

# Run with coverage
pytest --cov=app --cov-report=html
```

### 3. Run Linting Locally
```bash
# Run all linting tools
flake8 app/ tests/
black --check app/ tests/
isort --check-only app/ tests/
mypy app/
```

### 4. Run Security Checks Locally
```bash
# Run Bandit
bandit -r app/

# Run Safety
safety check

# Run Trivy (if installed)
trivy fs .
```

## Environment Variables

### Required Environment Variables
```bash
# Database
DATABASE_URL=postgresql://user:password@host:port/database
REDIS_URL=redis://host:port

# Security
SECRET_KEY=your-secret-key
JWT_SECRET=your-jwt-secret
ENCRYPTION_KEY=your-encryption-key

# Environment
ENVIRONMENT=development|staging|production
```

### Optional Environment Variables
```bash
# SMTP
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email
SMTP_PASSWORD=your-password

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true

# Features
AI_ENABLED=true
BLOCKCHAIN_ENABLED=true
SOCIAL_FEATURES_ENABLED=true
```

## Monitoring and Debugging

### 1. GitHub Actions Logs
- Check the Actions tab in GitHub repository
- Look for specific step failures
- Check service logs for database/Redis issues

### 2. Application Logs
```bash
# Check application logs in Kubernetes
kubectl logs -f deployment/opinion-market-api -n opinion-market

# Check service logs
kubectl logs -f service/opinion-market-api -n opinion-market
```

### 3. Health Check Endpoints
```bash
# Test health endpoints
curl http://localhost:8000/health
curl http://localhost:8000/ready
curl http://localhost:8000/metrics
```

## Best Practices

### 1. Code Quality
- Write tests for new features
- Follow PEP 8 style guidelines
- Use type hints
- Document functions and classes

### 2. Security
- Keep dependencies updated
- Use secrets for sensitive data
- Run security scans regularly
- Follow least privilege principle

### 3. Deployment
- Use semantic versioning
- Test in staging before production
- Monitor deployments
- Have rollback procedures

### 4. Monitoring
- Set up alerts for failures
- Monitor application metrics
- Track deployment success rates
- Log important events

## Troubleshooting Commands

### 1. Check Pipeline Status
```bash
# Check GitHub Actions status
gh run list

# Check specific run
gh run view <run-id>
```

### 2. Debug Local Issues
```bash
# Check Python environment
python --version
pip list

# Check Docker
docker --version
docker build -t test .

# Check Kubernetes
kubectl version
kubectl get pods -n opinion-market
```

### 3. Fix Common Issues
```bash
# Clear Python cache
find . -type d -name "__pycache__" -delete
find . -name "*.pyc" -delete

# Clear Docker cache
docker system prune -a

# Reset Kubernetes deployment
kubectl rollout restart deployment/opinion-market-api -n opinion-market
```

## Support

If you encounter issues not covered in this guide:

1. Check the GitHub Actions logs for detailed error messages
2. Review the application logs in Kubernetes
3. Test the failing component locally
4. Check the GitHub Issues page for similar problems
5. Create a new issue with detailed information about the problem

## Contributing

When contributing to the CI/CD pipeline:

1. Test changes locally first
2. Update this documentation if needed
3. Follow the existing patterns and conventions
4. Add appropriate error handling
5. Consider backward compatibility
