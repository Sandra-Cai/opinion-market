# 🛠️ CI/CD Troubleshooting Guide

## Overview

This guide provides comprehensive troubleshooting steps for the Opinion Market CI/CD pipeline. It covers common issues, their solutions, and preventive measures.

## Quick Reference

| Issue Type | Common Symptoms | Quick Fix |
|------------|----------------|-----------|
| **Build Failures** | Docker build errors, dependency issues | Check Dockerfile, update dependencies |
| **Test Failures** | Tests timing out, import errors | Verify test environment, check imports |
| **Deployment Issues** | Kubernetes errors, health check failures | Check kubeconfig, verify secrets |
| **Security Issues** | Bandit/Safety warnings, vulnerability alerts | Update dependencies, fix code issues |
| **Performance Issues** | Slow builds, timeouts | Optimize Docker layers, increase timeouts |

## 🔧 Common Issues and Solutions

### 1. Build Failures

#### Docker Build Failures

**Symptoms:**
- `docker build` commands failing
- Multi-stage build issues
- Platform-specific build errors

**Solutions:**

```bash
# Check Docker daemon status
docker info

# Clean Docker cache
docker system prune -a

# Test individual Dockerfiles
docker build -f Dockerfile.simple -t test-simple .
docker build -f Dockerfile.robust -t test-robust . --target base

# Check for platform issues
docker buildx ls
docker buildx create --use
```

**Prevention:**
- Use multi-stage builds efficiently
- Pin base image versions
- Optimize layer caching

#### Dependency Installation Failures

**Symptoms:**
- `pip install` errors
- Missing packages in requirements
- Version conflicts

**Solutions:**

```bash
# Update pip
python -m pip install --upgrade pip

# Install with verbose output
pip install -r requirements.txt -v

# Check for conflicts
pip check

# Install with no cache
pip install --no-cache-dir -r requirements.txt

# Try individual package installation
pip install fastapi uvicorn pytest httpx
```

**Prevention:**
- Pin dependency versions
- Use virtual environments
- Regular dependency updates

### 2. Test Failures

#### Import Errors

**Symptoms:**
- `ModuleNotFoundError`
- Import path issues
- Circular imports

**Solutions:**

```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Test imports individually
python -c "from app.main_simple import app"
python -c "import fastapi, uvicorn, pytest, httpx"

# Check for missing __init__.py files
find app/ -name "*.py" -exec dirname {} \; | sort -u | xargs -I {} touch {}/__init__.py
```

#### Test Timeout Issues

**Symptoms:**
- Tests hanging indefinitely
- Timeout errors in CI
- Slow test execution

**Solutions:**

```bash
# Run tests with timeout
timeout 300 pytest tests/ -v

# Run specific test categories
pytest tests/test_simple_app.py -v --tb=short
pytest tests/test_robust.py -v --tb=short

# Check for hanging processes
ps aux | grep python
```

**Prevention:**
- Set appropriate timeouts
- Use test fixtures properly
- Avoid blocking operations in tests

### 3. Deployment Issues

#### Kubernetes Configuration Issues

**Symptoms:**
- `kubectl` authentication errors
- Pod startup failures
- Service connectivity issues

**Solutions:**

```bash
# Check kubectl configuration
kubectl config current-context
kubectl cluster-info

# Verify secrets
kubectl get secrets -n opinion-market-staging
kubectl describe secret <secret-name> -n opinion-market-staging

# Check pod status
kubectl get pods -n opinion-market-staging
kubectl describe pod <pod-name> -n opinion-market-staging
kubectl logs <pod-name> -n opinion-market-staging
```

#### Health Check Failures

**Symptoms:**
- Health endpoints not responding
- Readiness probe failures
- Liveness probe failures

**Solutions:**

```bash
# Test health endpoints locally
curl -f http://localhost:8000/health
curl -f http://localhost:8000/ready

# Check application logs
kubectl logs -f deployment/opinion-market-api -n opinion-market-staging

# Verify service endpoints
kubectl get services -n opinion-market-staging
kubectl describe service opinion-market-api -n opinion-market-staging
```

### 4. Security Issues

#### Bandit Security Warnings

**Symptoms:**
- Security linting failures
- High-severity warnings
- Code quality issues

**Solutions:**

```bash
# Run Bandit with detailed output
bandit -r app/ -f json -o bandit-report.json -ll

# Fix specific issues
bandit -r app/ -f txt

# Exclude false positives
bandit -r app/ -f json -o bandit-report.json -ll --skip B101,B601
```

#### Safety Dependency Warnings

**Symptoms:**
- Known vulnerability alerts
- Outdated package warnings
- Security advisories

**Solutions:**

```bash
# Check for vulnerabilities
safety check

# Update vulnerable packages
pip install --upgrade <package-name>

# Check specific package
safety check --json --output safety-report.json
```

### 5. Performance Issues

#### Slow Build Times

**Symptoms:**
- Long Docker build times
- CI pipeline timeouts
- Resource exhaustion

**Solutions:**

```bash
# Optimize Docker builds
docker build --no-cache -f Dockerfile.robust -t test-robust .

# Use build cache
docker build --cache-from test-robust -f Dockerfile.robust -t test-robust .

# Check build context size
docker build --progress=plain -f Dockerfile.robust -t test-robust .
```

#### Memory Issues

**Symptoms:**
- Out of memory errors
- Process killed by system
- Slow performance

**Solutions:**

```bash
# Check memory usage
free -h
docker stats

# Limit Docker memory
docker run --memory=512m --memory-swap=1g <image>

# Monitor resource usage
htop
```

## 🔍 Debugging Commands

### System Diagnostics

```bash
# System information
uname -a
python --version
docker --version
kubectl version

# Resource usage
df -h
free -h
ps aux | head -20

# Network connectivity
ping github.com
curl -I https://api.github.com
```

### Application Diagnostics

```bash
# Python environment
python -c "import sys; print(sys.executable)"
python -c "import sys; print(sys.path)"
pip list

# Application health
python -c "from app.main_simple import app; print('App imported successfully')"
python -m uvicorn app.main_simple:app --host 0.0.0.0 --port 8000 &
curl -f http://localhost:8000/health
```

### CI/CD Pipeline Diagnostics

```bash
# Check GitHub Actions
gh run list --limit 10
gh run view <run-id>

# Check workflow files
find .github/workflows/ -name "*.yml" -exec yamllint {} \;

# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('.github/workflows/enhanced-ci-cd.yml'))"
```

## 📊 Monitoring and Alerting

### Health Check Scripts

```bash
# Run comprehensive health check
./scripts/health_check.sh

# Monitor CI/CD pipeline
./scripts/monitor_cicd.sh --single

# Test robust pipeline
./scripts/test_robust_pipeline.sh
```

### Log Analysis

```bash
# Check application logs
tail -f /var/log/opinion-market/app.log

# Check Docker logs
docker logs <container-name>

# Check Kubernetes logs
kubectl logs -f deployment/opinion-market-api -n opinion-market-staging
```

## 🚨 Emergency Procedures

### Pipeline Recovery

1. **Identify the Issue**
   ```bash
   gh run list --limit 5
   gh run view <failed-run-id>
   ```

2. **Check System Resources**
   ```bash
   df -h
   free -h
   docker system df
   ```

3. **Clean Up Resources**
   ```bash
   docker system prune -a
   kubectl delete pods --field-selector=status.phase=Failed
   ```

4. **Restart Services**
   ```bash
   kubectl rollout restart deployment/opinion-market-api -n opinion-market-staging
   ```

### Rollback Procedures

1. **Rollback Deployment**
   ```bash
   kubectl rollout undo deployment/opinion-market-api -n opinion-market-staging
   ```

2. **Rollback to Previous Image**
   ```bash
   kubectl set image deployment/opinion-market-api opinion-market-api=ghcr.io/your-repo/opinion-market:previous-tag -n opinion-market-staging
   ```

3. **Emergency Scale Down**
   ```bash
   kubectl scale deployment opinion-market-api --replicas=0 -n opinion-market-staging
   ```

## 📋 Preventive Measures

### Regular Maintenance

1. **Weekly Tasks**
   - Update dependencies
   - Review security reports
   - Check resource usage
   - Monitor pipeline performance

2. **Monthly Tasks**
   - Update base images
   - Review and optimize Dockerfiles
   - Update CI/CD workflows
   - Security audit

3. **Quarterly Tasks**
   - Infrastructure review
   - Performance optimization
   - Disaster recovery testing
   - Documentation updates

### Best Practices

1. **Code Quality**
   - Use pre-commit hooks
   - Regular code reviews
   - Automated testing
   - Security scanning

2. **Infrastructure**
   - Resource monitoring
   - Backup strategies
   - Disaster recovery plans
   - Security hardening

3. **CI/CD Pipeline**
   - Incremental improvements
   - Performance monitoring
   - Error handling
   - Documentation

## 📞 Support and Escalation

### Internal Support

1. **Level 1: Self-Service**
   - Check this troubleshooting guide
   - Run health check scripts
   - Review logs and metrics

2. **Level 2: Team Support**
   - Escalate to development team
   - Review code changes
   - Check infrastructure status

3. **Level 3: Expert Support**
   - Contact DevOps team
   - Infrastructure issues
   - Security incidents

### External Support

1. **GitHub Support**
   - Actions issues
   - Repository problems
   - API rate limits

2. **Cloud Provider Support**
   - Kubernetes issues
   - Infrastructure problems
   - Service outages

3. **Third-Party Services**
   - Docker Hub issues
   - PyPI problems
   - Security tool issues

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Kubernetes Troubleshooting](https://kubernetes.io/docs/tasks/debug-application-cluster/)
- [Python Security Best Practices](https://python-security.readthedocs.io/)

---

**Remember:** Always test fixes in a staging environment before applying to production!
