# 🚀 Performance Monitoring Guide

## Overview

The Opinion Market platform includes a comprehensive performance monitoring system that provides real-time insights into system performance, identifies bottlenecks, and offers optimization recommendations.

## Features

### 🔍 Real-Time Monitoring
- **System Metrics**: CPU, memory, disk, and network usage
- **Application Metrics**: Process-specific performance data
- **Custom Metrics**: Application-defined performance indicators
- **Execution Time Tracking**: Automatic timing of operations

### 📊 Performance Analytics
- **Historical Data**: Performance trends over time
- **Threshold Monitoring**: Configurable alert thresholds
- **Performance Scoring**: Overall system health scoring
- **Bottleneck Detection**: Automatic identification of performance issues

### 🚨 Intelligent Alerting
- **Multi-Level Alerts**: Low, medium, high, and critical severity levels
- **Smart Thresholds**: Dynamic threshold adjustment based on historical data
- **Alert Aggregation**: Prevents alert spam with intelligent grouping
- **Recovery Detection**: Automatic alert resolution when issues are fixed

### 💡 Optimization Recommendations
- **AI-Powered Analysis**: Machine learning-based performance analysis
- **Actionable Insights**: Specific recommendations for performance improvements
- **Impact Assessment**: Evaluation of potential performance gains
- **Implementation Guidance**: Step-by-step optimization instructions

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                 Performance Monitor                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Metrics   │  │   Alerts    │  │Recommendations│       │
│  │ Collection  │  │  Engine     │  │   Engine     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   System    │  │Application  │  │   Custom    │         │
│  │  Metrics    │  │  Metrics    │  │  Metrics    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### Enhanced Cache System

```
┌─────────────────────────────────────────────────────────────┐
│                 Enhanced Cache System                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │    LRU      │  │   TTL       │  │   Tag-based │        │
│  │ Eviction    │  │ Management  │  │ Invalidation│        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ Statistics  │  │  Decorator  │  │  Background │        │
│  │ Tracking    │  │   Support   │  │  Cleanup    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## API Endpoints

### Performance Monitoring

#### Get Metrics Summary
```http
GET /api/v1/performance/metrics/summary
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "cpu_usage": {
      "current": 45.2,
      "average": 42.1,
      "min": 35.0,
      "max": 55.0,
      "unit": "percent",
      "timestamp": "2024-01-15T10:30:00Z"
    },
    "memory_usage": {
      "current": 78.5,
      "average": 75.2,
      "min": 70.0,
      "max": 85.0,
      "unit": "percent",
      "timestamp": "2024-01-15T10:30:00Z"
    },
    "cache": {
      "hits": 1250,
      "misses": 150,
      "hit_rate": 89.3,
      "total_size": 52428800,
      "entry_count": 1250,
      "utilization": 62.5
    },
    "health_score": 87.5
  }
}
```

#### Get Performance Alerts
```http
GET /api/v1/performance/alerts?severity=high
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "alerts": [
      {
        "metric_name": "cpu_usage",
        "threshold": 80.0,
        "current_value": 85.2,
        "severity": "high",
        "message": "CPU usage is 85.2%, exceeding threshold of 80.0",
        "timestamp": "2024-01-15T10:30:00Z"
      }
    ],
    "count": 1,
    "severity_breakdown": {
      "low": 0,
      "medium": 2,
      "high": 1,
      "critical": 0
    }
  }
}
```

#### Get Performance Recommendations
```http
GET /api/v1/performance/recommendations?priority=high
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "recommendations": [
      {
        "category": "system",
        "priority": "high",
        "title": "High CPU Usage Detected",
        "description": "Average CPU usage is 85.1%. Consider optimizing CPU-intensive operations.",
        "impact": "Reduced system responsiveness and potential timeouts",
        "effort": "medium",
        "implementation": "Profile application code, optimize database queries, implement caching",
        "created_at": "2024-01-15T10:30:00Z"
      }
    ],
    "count": 1,
    "priority_breakdown": {
      "low": 0,
      "medium": 2,
      "high": 1,
      "critical": 0
    }
  }
}
```

### Cache Management

#### Get Cache Statistics
```http
GET /api/v1/performance/cache/stats
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "hits": 1250,
    "misses": 150,
    "evictions": 25,
    "hit_rate": 89.3,
    "total_size": 52428800,
    "entry_count": 1250,
    "max_size": 2000,
    "utilization": 62.5
  }
}
```

#### Clear Cache
```http
POST /api/v1/performance/cache/clear
Content-Type: application/json

{
  "tags": ["user", "market"]
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Cleared 150 cache entries with tags: ['user', 'market']",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Usage Examples

### Python Integration

#### Basic Performance Monitoring
```python
from app.core.performance_monitor import performance_monitor

# Start monitoring
await performance_monitor.start_monitoring(interval=30)

# Record custom metrics
performance_monitor.record_custom_metric("api_requests", 150, "count")
performance_monitor.record_custom_metric("response_time", 250, "ms")

# Measure execution time
async with performance_monitor.measure_execution_time("database_query"):
    result = await database.query("SELECT * FROM users")

# Get metrics summary
summary = performance_monitor.get_metrics_summary()
print(f"Current CPU usage: {summary['cpu_usage']['current']}%")
```

#### Enhanced Caching
```python
from app.core.enhanced_cache import enhanced_cache

# Basic caching
await enhanced_cache.set("user:123", user_data, ttl=3600)
user_data = await enhanced_cache.get("user:123")

# Tagged caching
await enhanced_cache.set("market:456", market_data, tags=["market", "active"])
await enhanced_cache.set("user:789", user_data, tags=["user", "premium"])

# Clear by tags
deleted_count = await enhanced_cache.delete_by_tags(["market"])

# Cache decorator
@enhanced_cache.cache_decorator(ttl=300, key_prefix="api")
async def expensive_calculation(x, y):
    # Expensive operation
    return x * y + complex_calculation()
```

### API Integration

#### Monitoring Dashboard
```javascript
// Get performance dashboard data
const response = await fetch('/api/v1/performance/health');
const dashboard = await response.json();

console.log(`System Health Score: ${dashboard.data.overall_security_score}`);
console.log(`Active Alerts: ${dashboard.data.active_alerts}`);
console.log(`Recommendations: ${dashboard.data.recommendations.length}`);
```

#### Cache Management
```javascript
// Get cache statistics
const stats = await fetch('/api/v1/performance/cache/stats');
const cacheStats = await stats.json();

console.log(`Cache Hit Rate: ${cacheStats.data.hit_rate}%`);
console.log(`Cache Utilization: ${cacheStats.data.utilization}%`);

// Clear cache
await fetch('/api/v1/performance/cache/clear', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ tags: ['user', 'market'] })
});
```

## Configuration

### Environment Variables

```bash
# Performance Monitoring
PERFORMANCE_MONITORING_ENABLED=true
PERFORMANCE_MONITORING_INTERVAL=30
PERFORMANCE_ALERT_THRESHOLDS='{"cpu_usage": 80, "memory_usage": 85}'

# Cache Configuration
CACHE_MAX_SIZE=2000
CACHE_DEFAULT_TTL=3600
CACHE_CLEANUP_INTERVAL=300
```

### Threshold Configuration

```python
# Customize alert thresholds
performance_monitor.thresholds = {
    "cpu_usage": {"warning": 70.0, "critical": 90.0},
    "memory_usage": {"warning": 80.0, "critical": 95.0},
    "disk_usage": {"warning": 85.0, "critical": 95.0},
    "api_response_time": {"warning": 1000.0, "critical": 5000.0},
    "database_query_time": {"warning": 500.0, "critical": 2000.0},
    "cache_hit_rate": {"warning": 70.0, "critical": 50.0},
    "error_rate": {"warning": 5.0, "critical": 10.0}
}
```

## Best Practices

### Performance Monitoring

1. **Set Appropriate Thresholds**
   - Base thresholds on historical data
   - Consider business impact of alerts
   - Use different thresholds for different environments

2. **Monitor Key Metrics**
   - System resources (CPU, memory, disk)
   - Application performance (response times, throughput)
   - Business metrics (user activity, revenue)

3. **Use Custom Metrics**
   - Track application-specific performance indicators
   - Monitor business-critical operations
   - Measure user experience metrics

### Caching Strategy

1. **Cache Key Design**
   - Use consistent naming conventions
   - Include version information for cache invalidation
   - Consider data access patterns

2. **TTL Management**
   - Set appropriate expiration times
   - Use shorter TTLs for frequently changing data
   - Implement cache warming for critical data

3. **Tag-Based Invalidation**
   - Group related data with tags
   - Use tags for bulk cache operations
   - Implement hierarchical cache invalidation

## Troubleshooting

### Common Issues

#### High CPU Usage Alerts
```bash
# Check system processes
top -p $(pgrep -f "opinion-market")

# Analyze performance metrics
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/performance/metrics/summary
```

#### Cache Performance Issues
```bash
# Check cache statistics
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/performance/cache/stats

# Clear cache if needed
curl -X POST -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/performance/cache/clear
```

#### Memory Leaks
```bash
# Monitor memory usage over time
watch -n 5 'curl -s -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/performance/metrics/summary | \
  jq ".data.memory_usage"'
```

### Performance Optimization

1. **Database Optimization**
   - Review slow queries
   - Add appropriate indexes
   - Optimize connection pooling

2. **Caching Optimization**
   - Increase cache hit rates
   - Optimize cache key strategies
   - Implement cache warming

3. **Application Optimization**
   - Profile code for bottlenecks
   - Optimize algorithms
   - Implement async processing

## Monitoring and Alerting

### Alert Channels

- **Email Notifications**: Critical alerts sent to administrators
- **Slack Integration**: Real-time alerts in team channels
- **Webhook Support**: Custom alert integrations
- **Dashboard Alerts**: Visual indicators in monitoring dashboards

### Alert Management

- **Alert Suppression**: Temporarily disable alerts during maintenance
- **Alert Escalation**: Automatic escalation for unresolved critical alerts
- **Alert Correlation**: Group related alerts to reduce noise
- **Alert History**: Track alert patterns and resolution times

## Security Considerations

- **Access Control**: Restrict performance monitoring access to authorized users
- **Data Privacy**: Ensure sensitive data is not exposed in metrics
- **Audit Logging**: Log all performance monitoring activities
- **Rate Limiting**: Implement rate limiting for monitoring endpoints

## Future Enhancements

- **Machine Learning**: Advanced anomaly detection and prediction
- **Distributed Monitoring**: Multi-instance performance tracking
- **Custom Dashboards**: User-configurable monitoring interfaces
- **Integration APIs**: Third-party monitoring tool integrations
