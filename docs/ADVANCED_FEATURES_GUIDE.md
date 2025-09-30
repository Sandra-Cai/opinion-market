# Advanced Features Guide

## Overview

The Opinion Market Platform now includes advanced features for enterprise-grade performance monitoring, intelligent analytics, and automated scaling. This guide covers the new capabilities and how to use them effectively.

## Table of Contents

1. [Advanced Analytics Engine](#advanced-analytics-engine)
2. [Auto-Scaling Manager](#auto-scaling-manager)
3. [Advanced Dashboard](#advanced-dashboard)
4. [Performance Optimizer V2](#performance-optimizer-v2)
5. [Intelligent Alerting System](#intelligent-alerting-system)
6. [API Endpoints](#api-endpoints)
7. [Configuration](#configuration)
8. [Troubleshooting](#troubleshooting)

## Advanced Analytics Engine

### Overview

The Advanced Analytics Engine provides machine learning-powered insights, predictions, and anomaly detection for your opinion market platform.

### Features

- **Predictive Analytics**: Forecast system performance and user behavior
- **Anomaly Detection**: Identify unusual patterns in system metrics
- **Insight Generation**: Automated analysis and recommendations
- **Real-time Processing**: Continuous analysis of system data

### Usage

#### Starting Analytics

```bash
curl -X POST "http://localhost:8000/api/v1/advanced-analytics/start-analytics" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

#### Getting Analytics Summary

```bash
curl -X GET "http://localhost:8000/api/v1/advanced-analytics/summary" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

#### Adding Data Points

```bash
curl -X POST "http://localhost:8000/api/v1/advanced-analytics/add-data-point" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "metric_name": "cpu_usage",
    "value": 75.5,
    "metadata": {"source": "system_monitor"}
  }'
```

### Configuration

The analytics engine can be configured through environment variables:

```bash
# Enable ML features
ANALYTICS_ML_ENABLED=true

# Set prediction horizons (minutes)
ANALYTICS_PREDICTION_HORIZONS=15,30,60,120

# Anomaly detection threshold
ANALYTICS_ANOMALY_THRESHOLD=2.5
```

## Auto-Scaling Manager

### Overview

The Auto-Scaling Manager automatically adjusts system resources based on performance metrics and predictions.

### Features

- **Intelligent Scaling**: ML-powered scaling decisions
- **Multiple Policies**: CPU, memory, response time, and throughput scaling
- **Predictive Scaling**: Scale based on predicted load
- **Cost Optimization**: Balance performance and cost

### Usage

#### Starting Auto-Scaling

```bash
curl -X POST "http://localhost:8000/api/v1/auto-scaling/start-scaling" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

#### Creating Scaling Policies

```bash
curl -X POST "http://localhost:8000/api/v1/auto-scaling/policies" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "custom_cpu_scaling",
    "metric_name": "system.cpu_usage",
    "scale_up_threshold": 85.0,
    "scale_down_threshold": 30.0,
    "min_instances": 2,
    "max_instances": 20,
    "cooldown_period": 300
  }'
```

#### Manual Scaling

```bash
curl -X POST "http://localhost:8000/api/v1/auto-scaling/scale-manual" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "target_instances": 5,
    "reason": "Expected high load"
  }'
```

### Default Policies

The system comes with pre-configured policies:

- **CPU Scaling**: Scale when CPU usage exceeds 80%
- **Memory Scaling**: Scale when memory usage exceeds 85%
- **Response Time Scaling**: Scale when response time exceeds 100ms
- **Throughput Scaling**: Scale when requests per second exceed 1000

## Advanced Dashboard

### Overview

The Advanced Dashboard provides real-time monitoring with AI-powered insights and predictive analytics.

### Features

- **Real-time Metrics**: Live system performance data
- **Interactive Charts**: Historical performance visualization
- **AI Insights**: Automated performance analysis
- **WebSocket Updates**: Real-time data streaming
- **Alert Integration**: Visual alert notifications

### Accessing the Dashboard

Navigate to: `http://localhost:8000/api/v1/advanced-dashboard/dashboard/advanced`

### Dashboard Components

#### Metrics Cards
- CPU Usage with trend indicators
- Memory Usage with trend indicators
- Cache Hit Rate with trend indicators
- Response Time with trend indicators
- Throughput with trend indicators
- Health Score with trend indicators

#### Charts
- System Performance Over Time
- Cache Performance Analysis

#### AI Insights Panel
- Performance warnings and recommendations
- Optimization opportunities
- System health assessments

#### Alerts Panel
- Active system alerts
- Severity-based color coding
- Timestamp information

## Performance Optimizer V2

### Overview

Performance Optimizer V2 provides advanced performance optimization with intelligent resource management and automated tuning.

### Features

- **Intelligent Caching**: Advanced cache optimization strategies
- **Memory Management**: Automated garbage collection and cleanup
- **Resource Optimization**: Dynamic resource allocation
- **Performance Baselines**: Adaptive performance targets
- **Multi-Strategy Support**: Aggressive, balanced, and conservative modes

### Usage

#### Setting Optimization Strategy

```python
from app.core.performance_optimizer_v2 import performance_optimizer_v2

# Set aggressive optimization
performance_optimizer_v2.set_optimization_strategy("aggressive")

# Set balanced optimization (default)
performance_optimizer_v2.set_optimization_strategy("balanced")

# Set conservative optimization
performance_optimizer_v2.set_optimization_strategy("conservative")
```

#### Getting Performance Summary

```python
summary = performance_optimizer_v2.get_performance_summary()
print(f"Performance Score: {summary['performance_score']}")
print(f"Current Strategy: {summary['current_strategy']}")
```

### Optimization Strategies

#### Aggressive
- Cache cleanup every 30 seconds
- GC trigger at 70% memory usage
- Optimization every 60 seconds

#### Balanced (Default)
- Cache cleanup every 2 minutes
- GC trigger at 80% memory usage
- Optimization every 5 minutes

#### Conservative
- Cache cleanup every 5 minutes
- GC trigger at 90% memory usage
- Optimization every 10 minutes

## Intelligent Alerting System

### Overview

The Intelligent Alerting System provides advanced alerting with machine learning capabilities and multiple notification channels.

### Features

- **Smart Alerting**: ML-powered alert generation
- **Multiple Channels**: Email, webhook, Slack, SMS
- **Alert Escalation**: Automatic severity escalation
- **Suppression Rules**: Prevent alert fatigue
- **Custom Rules**: Flexible alert configuration

### Usage

#### Creating Alert Rules

```bash
curl -X POST "http://localhost:8000/api/v1/intelligent-alerting/rules" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "custom_alert",
    "name": "Custom Performance Alert",
    "description": "Alert when custom metric exceeds threshold",
    "metric_name": "custom_metric",
    "condition": "gt",
    "threshold": 90.0,
    "severity": "critical",
    "notification_channels": ["email", "slack"]
  }'
```

#### Acknowledging Alerts

```bash
curl -X POST "http://localhost:8000/api/v1/intelligent-alerting/alerts/{alert_id}/acknowledge" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "acknowledged_by": "admin@example.com"
  }'
```

#### Testing Alerts

```bash
curl -X POST "http://localhost:8000/api/v1/intelligent-alerting/test-alert" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "rule_id": "cpu_high"
  }'
```

### Default Alert Rules

- **High CPU Usage**: Warning at 80%, Critical at 95%
- **High Memory Usage**: Warning at 85%
- **Slow Response Time**: Warning at 200ms
- **Low Cache Hit Rate**: Warning below 70%
- **Low Disk Space**: Critical at 90%

### Notification Channels

#### Email
```json
{
  "id": "email",
  "name": "Email Notifications",
  "type": "email",
  "config": {
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "username": "alerts@yourcompany.com",
    "password": "your_password",
    "from_email": "alerts@yourcompany.com",
    "to_emails": ["admin@yourcompany.com"]
  }
}
```

#### Slack
```json
{
  "id": "slack",
  "name": "Slack Notifications",
  "type": "slack",
  "config": {
    "webhook_url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
    "channel": "#alerts",
    "username": "AlertBot"
  }
}
```

#### Webhook
```json
{
  "id": "webhook",
  "name": "Webhook Notifications",
  "type": "webhook",
  "config": {
    "url": "https://your-webhook-endpoint.com/alerts",
    "timeout": 10,
    "retry_attempts": 3
  }
}
```

## API Endpoints

### Advanced Analytics

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/advanced-analytics/start-analytics` | POST | Start analytics engine |
| `/api/v1/advanced-analytics/stop-analytics` | POST | Stop analytics engine |
| `/api/v1/advanced-analytics/summary` | GET | Get analytics summary |
| `/api/v1/advanced-analytics/predictions` | GET | Get predictions |
| `/api/v1/advanced-analytics/insights` | GET | Get insights |
| `/api/v1/advanced-analytics/anomalies` | GET | Get anomalies |
| `/api/v1/advanced-analytics/add-data-point` | POST | Add data point |

### Auto-Scaling

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/auto-scaling/start-scaling` | POST | Start auto-scaling |
| `/api/v1/auto-scaling/stop-scaling` | POST | Stop auto-scaling |
| `/api/v1/auto-scaling/summary` | GET | Get scaling summary |
| `/api/v1/auto-scaling/policies` | GET/POST | Manage scaling policies |
| `/api/v1/auto-scaling/decisions` | GET | Get scaling decisions |
| `/api/v1/auto-scaling/scale-manual` | POST | Manual scaling |
| `/api/v1/auto-scaling/recommendations` | GET | Get scaling recommendations |

### Advanced Dashboard

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/advanced-dashboard/dashboard/advanced` | GET | Get dashboard HTML |
| `/api/v1/advanced-dashboard/ws` | WebSocket | Real-time updates |
| `/api/v1/advanced-dashboard/dashboard/summary` | GET | Get dashboard summary |

### Intelligent Alerting

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/intelligent-alerting/start-alerting` | POST | Start alerting system |
| `/api/v1/intelligent-alerting/stop-alerting` | POST | Stop alerting system |
| `/api/v1/intelligent-alerting/summary` | GET | Get alerting summary |
| `/api/v1/intelligent-alerting/alerts` | GET | Get alerts |
| `/api/v1/intelligent-alerting/rules` | GET/POST/PUT/DELETE | Manage alert rules |
| `/api/v1/intelligent-alerting/channels` | GET/POST | Manage notification channels |
| `/api/v1/intelligent-alerting/test-alert` | POST | Test alert rule |

## Configuration

### Environment Variables

```bash
# Analytics Configuration
ANALYTICS_ML_ENABLED=true
ANALYTICS_PREDICTION_HORIZONS=15,30,60,120
ANALYTICS_ANOMALY_THRESHOLD=2.5

# Auto-Scaling Configuration
AUTO_SCALING_ENABLED=true
AUTO_SCALING_MIN_INSTANCES=1
AUTO_SCALING_MAX_INSTANCES=10
AUTO_SCALING_COOLDOWN_PERIOD=300

# Performance Optimization
PERFORMANCE_OPTIMIZATION_STRATEGY=balanced
PERFORMANCE_OPTIMIZATION_ENABLED=true

# Alerting Configuration
ALERTING_ENABLED=true
ALERTING_EVALUATION_INTERVAL=30
ALERTING_ESCALATION_TIMEOUT=1800

# Notification Channels
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=alerts@yourcompany.com
SMTP_PASSWORD=your_password
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
```

### Configuration Files

#### Performance Optimization Config

```yaml
# config/performance.yaml
optimization:
  strategy: "balanced"  # aggressive, balanced, conservative
  enabled: true
  
thresholds:
  cpu_usage:
    warning: 70
    critical: 85
  memory_usage:
    warning: 75
    critical: 90
  response_time:
    warning: 100
    critical: 200
```

#### Alerting Config

```yaml
# config/alerting.yaml
alerting:
  enabled: true
  evaluation_interval: 30
  escalation_timeout: 1800
  
channels:
  email:
    enabled: true
    smtp_server: "smtp.gmail.com"
    smtp_port: 587
  slack:
    enabled: true
    webhook_url: "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
```

## Troubleshooting

### Common Issues

#### Analytics Engine Not Starting

**Problem**: Analytics engine fails to start
**Solution**: 
1. Check if ML libraries are installed: `pip install scikit-learn`
2. Verify Redis connection if using Redis
3. Check logs for specific error messages

#### Auto-Scaling Not Working

**Problem**: Auto-scaling not triggering
**Solution**:
1. Verify scaling policies are enabled
2. Check if metrics are being collected
3. Ensure cooldown periods are not blocking scaling
4. Verify min/max instance limits

#### Dashboard Not Loading

**Problem**: Advanced dashboard shows loading or errors
**Solution**:
1. Check WebSocket connection
2. Verify all required services are running
3. Check browser console for JavaScript errors
4. Ensure proper authentication

#### Alerts Not Sending

**Problem**: Alerts not being sent to notification channels
**Solution**:
1. Verify notification channel configuration
2. Check network connectivity for webhooks
3. Verify email SMTP settings
4. Check alert rule severity filters

### Performance Issues

#### High Memory Usage

**Symptoms**: Memory usage consistently above 85%
**Solutions**:
1. Enable aggressive memory cleanup
2. Reduce cache size limits
3. Increase garbage collection frequency
4. Review application memory leaks

#### Slow Response Times

**Symptoms**: API response times above 200ms
**Solutions**:
1. Enable auto-scaling for response time
2. Optimize database queries
3. Increase cache hit rates
4. Review application bottlenecks

#### Cache Performance Issues

**Symptoms**: Low cache hit rates below 70%
**Solutions**:
1. Increase cache size
2. Review cache eviction policies
3. Optimize cache key strategies
4. Enable cache compression

### Monitoring and Debugging

#### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Check System Health

```bash
# Check all services status
curl -X GET "http://localhost:8000/health"

# Check performance summary
curl -X GET "http://localhost:8000/api/v1/advanced-analytics/summary" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Check alerting status
curl -X GET "http://localhost:8000/api/v1/intelligent-alerting/health" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

#### View Logs

```bash
# Application logs
tail -f logs/app.log

# Error logs
tail -f logs/error.log

# Performance logs
grep "performance" logs/app.log
```

## Best Practices

### Performance Optimization

1. **Start with Balanced Strategy**: Use balanced optimization as default
2. **Monitor Baselines**: Establish performance baselines before optimization
3. **Gradual Changes**: Make incremental changes and monitor impact
4. **Regular Reviews**: Review performance metrics weekly

### Alerting

1. **Set Appropriate Thresholds**: Avoid too sensitive or too loose thresholds
2. **Use Severity Levels**: Distinguish between info, warning, critical, emergency
3. **Configure Escalation**: Set up proper escalation procedures
4. **Test Alerts**: Regularly test alert rules and notification channels

### Auto-Scaling

1. **Conservative Limits**: Start with conservative min/max instance limits
2. **Monitor Costs**: Track scaling costs and optimize policies
3. **Predictive Scaling**: Enable predictive scaling for better performance
4. **Regular Tuning**: Adjust scaling policies based on usage patterns

### Analytics

1. **Data Quality**: Ensure high-quality metric data
2. **Regular Model Updates**: Retrain ML models regularly
3. **Anomaly Investigation**: Investigate detected anomalies promptly
4. **Insight Action**: Act on generated insights and recommendations

## Support

For additional support and questions:

1. Check the troubleshooting section above
2. Review application logs for error details
3. Consult the API documentation
4. Contact the development team

## Changelog

### Version 2.0.0
- Added Advanced Analytics Engine with ML capabilities
- Implemented Auto-Scaling Manager with predictive scaling
- Created Advanced Dashboard with real-time monitoring
- Added Performance Optimizer V2 with intelligent optimization
- Implemented Intelligent Alerting System with multiple channels
- Enhanced API documentation and user guides
