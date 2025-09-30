# Opinion Market API Reference

## Overview

The Opinion Market API provides comprehensive endpoints for managing prediction markets, user interactions, and advanced system features. This reference covers all available endpoints with detailed examples.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

All endpoints require authentication using Bearer tokens:

```bash
Authorization: Bearer YOUR_JWT_TOKEN
```

## Response Format

All API responses follow this format:

```json
{
  "success": true,
  "data": { ... },
  "timestamp": 1640995200.0,
  "message": "Optional message"
}
```

## Error Handling

Errors are returned with appropriate HTTP status codes:

```json
{
  "detail": "Error description",
  "status_code": 400
}
```

## Advanced Analytics API

### Start Analytics Engine

Start the advanced analytics engine.

```http
POST /advanced-analytics/start-analytics
```

**Response:**
```json
{
  "success": true,
  "message": "Advanced analytics engine started",
  "timestamp": 1640995200.0
}
```

### Stop Analytics Engine

Stop the advanced analytics engine.

```http
POST /advanced-analytics/stop-analytics
```

### Get Analytics Summary

Get comprehensive analytics summary.

```http
GET /advanced-analytics/summary
```

**Response:**
```json
{
  "success": true,
  "data": {
    "timestamp": "2024-01-01T12:00:00",
    "analytics_active": true,
    "data_points_collected": 1500,
    "models_trained": 5,
    "recent_predictions": 12,
    "recent_insights": 8,
    "recent_anomalies": 2,
    "ml_available": true
  }
}
```

### Get Predictions

Get predictions from the analytics engine.

```http
GET /advanced-analytics/predictions?metric_name=cpu_usage&horizon=30&limit=10
```

**Query Parameters:**
- `metric_name` (optional): Filter by metric name
- `horizon` (optional): Filter by time horizon in minutes
- `limit` (optional): Limit number of results (default: 50)

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "metric_name": "cpu_usage",
      "predicted_value": 75.5,
      "confidence": 0.85,
      "time_horizon": 30,
      "trend": "increasing",
      "factors": {
        "historical_values": 0.7,
        "time_features": 0.2,
        "trend": 0.1
      },
      "created_at": "2024-01-01T12:00:00"
    }
  ]
}
```

### Get Insights

Get analytical insights.

```http
GET /advanced-analytics/insights?insight_type=performance&impact=high&limit=20
```

**Query Parameters:**
- `insight_type` (optional): Filter by insight type
- `impact` (optional): Filter by impact level
- `limit` (optional): Limit number of results (default: 20)

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "insight_type": "performance",
      "title": "Excellent System Performance",
      "description": "System performance score averaging 95.2/100",
      "confidence": 0.95,
      "impact": "high",
      "recommendations": [
        "Continue current optimization strategies",
        "Monitor for any degradation"
      ],
      "data_points": {
        "performance_score": 95.2,
        "sample_size": 100
      },
      "created_at": "2024-01-01T12:00:00"
    }
  ]
}
```

### Get Anomalies

Get detected anomalies.

```http
GET /advanced-analytics/anomalies?metric_name=cpu_usage&severity=high&limit=50
```

**Query Parameters:**
- `metric_name` (optional): Filter by metric name
- `severity` (optional): Filter by severity level
- `limit` (optional): Limit number of results (default: 50)

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "metric_name": "cpu_usage",
      "anomaly_score": 3.2,
      "is_anomaly": true,
      "expected_value": 50.0,
      "actual_value": 95.0,
      "severity": "high",
      "description": "Value 95.0 is 3.2 standard deviations from mean 50.0",
      "timestamp": "2024-01-01T12:00:00"
    }
  ]
}
```

### Add Data Point

Add a data point for analysis.

```http
POST /advanced-analytics/add-data-point
```

**Request Body:**
```json
{
  "metric_name": "cpu_usage",
  "value": 75.5,
  "metadata": {
    "source": "system_monitor",
    "host": "server-01"
  }
}
```

**Response:**
```json
{
  "success": true,
  "message": "Data point added for cpu_usage",
  "data": {
    "metric_name": "cpu_usage",
    "value": 75.5,
    "metadata": {
      "source": "system_monitor",
      "host": "server-01"
    },
    "timestamp": "2024-01-01T12:00:00"
  }
}
```

## Auto-Scaling API

### Start Auto-Scaling

Start the auto-scaling manager.

```http
POST /auto-scaling/start-scaling
```

### Stop Auto-Scaling

Stop the auto-scaling manager.

```http
POST /auto-scaling/stop-scaling
```

### Get Scaling Summary

Get comprehensive auto-scaling summary.

```http
GET /auto-scaling/summary
```

**Response:**
```json
{
  "success": true,
  "data": {
    "timestamp": "2024-01-01T12:00:00",
    "scaling_active": true,
    "current_instances": 3,
    "target_instances": 3,
    "policies": {
      "cpu_scaling": {
        "enabled": true,
        "metric_name": "system.cpu_usage",
        "scale_up_threshold": 80.0,
        "scale_down_threshold": 30.0,
        "min_instances": 1,
        "max_instances": 10
      }
    },
    "recent_decisions": [
      {
        "action": "scale_up",
        "target_instances": 3,
        "current_instances": 2,
        "reason": "system.cpu_usage at 85.0 exceeds threshold 80.0",
        "confidence": 0.9,
        "created_at": "2024-01-01T12:00:00"
      }
    ]
  }
}
```

### Get Scaling Policies

Get all scaling policies.

```http
GET /auto-scaling/policies
```

**Response:**
```json
{
  "success": true,
  "data": {
    "policies": {
      "cpu_scaling": {
        "name": "CPU Scaling",
        "metric_name": "system.cpu_usage",
        "scale_up_threshold": 80.0,
        "scale_down_threshold": 30.0,
        "min_instances": 1,
        "max_instances": 10,
        "cooldown_period": 300,
        "enabled": true
      }
    },
    "total_policies": 4
  }
}
```

### Create Scaling Policy

Create a new scaling policy.

```http
POST /auto-scaling/policies
```

**Request Body:**
```json
{
  "name": "custom_memory_scaling",
  "metric_name": "system.memory_usage",
  "scale_up_threshold": 85.0,
  "scale_down_threshold": 40.0,
  "min_instances": 2,
  "max_instances": 20,
  "cooldown_period": 300,
  "enabled": true
}
```

**Response:**
```json
{
  "success": true,
  "message": "Scaling policy 'custom_memory_scaling' created successfully",
  "data": {
    "name": "custom_memory_scaling",
    "metric_name": "system.memory_usage",
    "scale_up_threshold": 85.0,
    "scale_down_threshold": 40.0,
    "min_instances": 2,
    "max_instances": 20,
    "cooldown_period": 300,
    "enabled": true
  }
}
```

### Update Scaling Policy

Update an existing scaling policy.

```http
PUT /auto-scaling/policies/{policy_name}
```

**Request Body:**
```json
{
  "scale_up_threshold": 90.0,
  "enabled": false
}
```

### Delete Scaling Policy

Delete a scaling policy.

```http
DELETE /auto-scaling/policies/{policy_name}
```

### Get Scaling Decisions

Get scaling decisions.

```http
GET /auto-scaling/decisions?action=scale_up&limit=50
```

**Query Parameters:**
- `action` (optional): Filter by action type
- `limit` (optional): Limit number of results (default: 50)

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "action": "scale_up",
      "target_instances": 3,
      "current_instances": 2,
      "reason": "system.cpu_usage at 85.0 exceeds threshold 80.0",
      "confidence": 0.9,
      "metrics": {
        "cpu_usage": 85.0,
        "memory_usage": 60.0
      },
      "predicted_impact": {
        "cost_change": 0.1,
        "performance_change": 25.0
      },
      "created_at": "2024-01-01T12:00:00"
    }
  ]
}
```

### Manual Scaling

Manually trigger scaling to a specific number of instances.

```http
POST /auto-scaling/scale-manual
```

**Request Body:**
```json
{
  "target_instances": 5,
  "reason": "Expected high load during event"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Successfully scaled to 5 instances",
  "data": {
    "target_instances": 5,
    "previous_instances": 3,
    "reason": "Expected high load during event"
  }
}
```

### Get Scaling Recommendations

Get intelligent scaling recommendations.

```http
GET /auto-scaling/recommendations
```

**Response:**
```json
{
  "success": true,
  "data": {
    "recommendations": [
      {
        "type": "scale_up",
        "policy": "cpu_scaling",
        "metric": "system.cpu_usage",
        "current_value": 85.0,
        "threshold": 80.0,
        "recommended_instances": 4,
        "reason": "system.cpu_usage exceeds scale-up threshold",
        "priority": "high"
      }
    ],
    "total_recommendations": 1,
    "current_instances": 3
  }
}
```

## Advanced Dashboard API

### Get Dashboard

Get the advanced real-time performance dashboard.

```http
GET /advanced-dashboard/dashboard/advanced
```

**Response:** HTML page with real-time dashboard

### Dashboard WebSocket

WebSocket endpoint for real-time dashboard updates.

```http
WS /advanced-dashboard/ws
```

**Message Format:**
```json
{
  "timestamp": "2024-01-01T12:00:00",
  "metrics": {
    "cpu_usage": 75.5,
    "memory_usage": 60.0,
    "cache_hit_rate": 85.0,
    "response_time": 45.0,
    "throughput": 1200.0
  },
  "performance_score": 88.5,
  "trends": {
    "cpu_usage": "stable",
    "memory_usage": "increasing",
    "cache_hit_rate": "stable"
  },
  "ai_insights": [
    {
      "type": "Performance Warning",
      "message": "High CPU usage detected. Consider scaling or optimization.",
      "confidence": 0.9
    }
  ],
  "alerts": [
    {
      "severity": "warning",
      "title": "High CPU Usage",
      "message": "CPU usage is at 85.0%",
      "timestamp": "2024-01-01T12:00:00"
    }
  ]
}
```

### Get Dashboard Summary

Get dashboard summary data.

```http
GET /advanced-dashboard/dashboard/summary
```

**Response:**
```json
{
  "success": true,
  "data": {
    "timestamp": "2024-01-01T12:00:00",
    "metrics": {
      "cpu_usage": 75.5,
      "memory_usage": 60.0,
      "cache_hit_rate": 85.0,
      "response_time": 45.0,
      "throughput": 1200.0
    },
    "performance_score": 88.5,
    "trends": {
      "cpu_usage": "stable",
      "memory_usage": "increasing"
    },
    "ai_insights": [...],
    "alerts": [...],
    "active_connections": 3
  }
}
```

## Intelligent Alerting API

### Start Alerting System

Start the intelligent alerting system.

```http
POST /intelligent-alerting/start-alerting
```

### Stop Alerting System

Stop the intelligent alerting system.

```http
POST /intelligent-alerting/stop-alerting
```

### Get Alerting Summary

Get comprehensive alerting system summary.

```http
GET /intelligent-alerting/summary
```

**Response:**
```json
{
  "success": true,
  "data": {
    "timestamp": "2024-01-01T12:00:00",
    "alerting_active": true,
    "last_evaluation": "2024-01-01T12:00:00",
    "active_alerts": {
      "total": 2,
      "by_severity": {
        "warning": 1,
        "critical": 1
      }
    },
    "recent_alerts": 5,
    "total_rules": 6,
    "enabled_rules": 5,
    "notification_channels": 3,
    "stats": {
      "alerts_triggered": 25,
      "alerts_resolved": 23,
      "false_positives": 2,
      "notifications_sent": 50,
      "average_resolution_time": 1800.0
    }
  }
}
```

### Get Alerts

Get alerts with optional filtering.

```http
GET /intelligent-alerting/alerts?status=active&severity=critical&limit=20
```

**Query Parameters:**
- `status` (optional): Filter by alert status
- `severity` (optional): Filter by severity level
- `limit` (optional): Limit number of results (default: 50)

**Response:**
```json
{
  "success": true,
  "data": {
    "alerts": [
      {
        "id": "cpu_high_1640995200",
        "rule_id": "cpu_high",
        "title": "High CPU Usage",
        "message": "CPU usage exceeds threshold. Current value: 85.0, Threshold: 80.0",
        "severity": "warning",
        "status": "active",
        "metric_name": "cpu_usage",
        "current_value": 85.0,
        "threshold_value": 80.0,
        "triggered_at": "2024-01-01T12:00:00",
        "acknowledged_at": null,
        "acknowledged_by": null,
        "metadata": {
          "rule_name": "High CPU Usage",
          "condition": "gt",
          "tags": ["system", "performance"]
        }
      }
    ],
    "total_count": 1,
    "active_count": 1
  }
}
```

### Get Alert Rules

Get alert rules.

```http
GET /intelligent-alerting/rules?enabled_only=true
```

**Query Parameters:**
- `enabled_only` (optional): Only return enabled rules

**Response:**
```json
{
  "success": true,
  "data": {
    "rules": [
      {
        "id": "cpu_high",
        "name": "High CPU Usage",
        "description": "CPU usage exceeds threshold",
        "metric_name": "cpu_usage",
        "condition": "gt",
        "threshold": 80.0,
        "severity": "warning",
        "enabled": true,
        "cooldown_period": 300,
        "notification_channels": ["email", "webhook"],
        "tags": ["system", "performance"],
        "created_at": "2024-01-01T12:00:00"
      }
    ],
    "total_count": 6
  }
}
```

### Create Alert Rule

Create a new alert rule.

```http
POST /intelligent-alerting/rules
```

**Request Body:**
```json
{
  "id": "custom_alert",
  "name": "Custom Performance Alert",
  "description": "Alert when custom metric exceeds threshold",
  "metric_name": "custom_metric",
  "condition": "gt",
  "threshold": 90.0,
  "severity": "critical",
  "enabled": true,
  "cooldown_period": 300,
  "notification_channels": ["email", "slack"],
  "tags": ["custom", "performance"]
}
```

**Response:**
```json
{
  "success": true,
  "message": "Alert rule 'Custom Performance Alert' created successfully",
  "data": {
    "id": "custom_alert",
    "name": "Custom Performance Alert",
    "metric_name": "custom_metric",
    "severity": "critical",
    "enabled": true
  }
}
```

### Update Alert Rule

Update an existing alert rule.

```http
PUT /intelligent-alerting/rules/{rule_id}
```

**Request Body:**
```json
{
  "threshold": 95.0,
  "enabled": false
}
```

### Delete Alert Rule

Delete an alert rule.

```http
DELETE /intelligent-alerting/rules/{rule_id}
```

### Acknowledge Alert

Acknowledge an alert.

```http
POST /intelligent-alerting/alerts/{alert_id}/acknowledge
```

**Request Body:**
```json
{
  "acknowledged_by": "admin@example.com"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Alert 'cpu_high_1640995200' acknowledged by admin@example.com",
  "timestamp": 1640995200.0
}
```

### Get Notification Channels

Get notification channels.

```http
GET /intelligent-alerting/channels
```

**Response:**
```json
{
  "success": true,
  "data": {
    "channels": [
      {
        "id": "email",
        "name": "Email Notifications",
        "type": "email",
        "enabled": true,
        "severity_filter": ["warning", "critical", "emergency"],
        "config": {
          "smtp_server": "smtp.gmail.com",
          "smtp_port": 587,
          "from_email": "alerts@opinionmarket.com",
          "to_emails": ["admin@opinionmarket.com"]
        }
      }
    ],
    "total_count": 3
  }
}
```

### Create Notification Channel

Create a new notification channel.

```http
POST /intelligent-alerting/channels
```

**Request Body:**
```json
{
  "id": "custom_webhook",
  "name": "Custom Webhook",
  "type": "webhook",
  "enabled": true,
  "severity_filter": ["critical", "emergency"],
  "config": {
    "url": "https://your-webhook-endpoint.com/alerts",
    "timeout": 10,
    "retry_attempts": 3
  }
}
```

### Test Alert

Test an alert rule by triggering a test alert.

```http
POST /intelligent-alerting/test-alert
```

**Request Body:**
```json
{
  "rule_id": "cpu_high"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Test alert triggered for rule 'High CPU Usage'",
  "data": {
    "rule_id": "cpu_high",
    "rule_name": "High CPU Usage",
    "test_value": 90.0,
    "threshold": 80.0
  }
}
```

### Get Alerting Stats

Get alerting system statistics.

```http
GET /intelligent-alerting/stats
```

**Response:**
```json
{
  "success": true,
  "data": {
    "alerts_triggered": 25,
    "alerts_resolved": 23,
    "false_positives": 2,
    "notifications_sent": 50,
    "average_resolution_time": 1800.0,
    "active_alerts": 2,
    "total_rules": 6,
    "enabled_rules": 5
  }
}
```

### Get Alerting Health

Get alerting system health status.

```http
GET /intelligent-alerting/health
```

**Response:**
```json
{
  "success": true,
  "data": {
    "alerting_active": true,
    "last_evaluation": "2024-01-01T12:00:00",
    "active_alerts": 2,
    "enabled_rules": 5,
    "notification_channels": 3,
    "status": "healthy"
  }
}
```

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid input data |
| 401 | Unauthorized - Invalid or missing authentication |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Resource not found |
| 409 | Conflict - Resource already exists |
| 422 | Unprocessable Entity - Validation error |
| 500 | Internal Server Error - Server error |

## Rate Limiting

API endpoints are rate-limited to prevent abuse:

- **Default**: 100 requests per minute per IP
- **Authentication endpoints**: 10 requests per minute per IP
- **Analytics endpoints**: 50 requests per minute per user
- **Scaling endpoints**: 20 requests per minute per user

Rate limit headers are included in responses:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995260
```

## Webhooks

The system supports webhooks for real-time notifications:

### Webhook Payload Format

```json
{
  "event": "alert.triggered",
  "timestamp": "2024-01-01T12:00:00",
  "data": {
    "alert_id": "cpu_high_1640995200",
    "rule_id": "cpu_high",
    "title": "High CPU Usage",
    "severity": "warning",
    "metric_name": "cpu_usage",
    "current_value": 85.0,
    "threshold_value": 80.0
  }
}
```

### Supported Events

- `alert.triggered` - New alert triggered
- `alert.acknowledged` - Alert acknowledged
- `alert.resolved` - Alert resolved
- `scaling.action` - Scaling action taken
- `analytics.insight` - New insight generated

## SDKs and Libraries

### Python SDK

```python
from opinion_market_sdk import OpinionMarketClient

client = OpinionMarketClient(
    base_url="http://localhost:8000/api/v1",
    token="your_jwt_token"
)

# Get analytics summary
summary = client.analytics.get_summary()

# Create scaling policy
policy = client.scaling.create_policy({
    "name": "custom_scaling",
    "metric_name": "cpu_usage",
    "scale_up_threshold": 80.0,
    "scale_down_threshold": 30.0
})

# Get alerts
alerts = client.alerting.get_alerts(status="active")
```

### JavaScript SDK

```javascript
import { OpinionMarketClient } from 'opinion-market-sdk';

const client = new OpinionMarketClient({
  baseUrl: 'http://localhost:8000/api/v1',
  token: 'your_jwt_token'
});

// Get analytics summary
const summary = await client.analytics.getSummary();

// Create scaling policy
const policy = await client.scaling.createPolicy({
  name: 'custom_scaling',
  metric_name: 'cpu_usage',
  scale_up_threshold: 80.0,
  scale_down_threshold: 30.0
});

// Get alerts
const alerts = await client.alerting.getAlerts({ status: 'active' });
```

## Examples

### Complete Workflow Example

```bash
# 1. Start analytics engine
curl -X POST "http://localhost:8000/api/v1/advanced-analytics/start-analytics" \
  -H "Authorization: Bearer YOUR_TOKEN"

# 2. Add some data points
curl -X POST "http://localhost:8000/api/v1/advanced-analytics/add-data-point" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"metric_name": "cpu_usage", "value": 75.5}'

# 3. Get insights
curl -X GET "http://localhost:8000/api/v1/advanced-analytics/insights" \
  -H "Authorization: Bearer YOUR_TOKEN"

# 4. Start auto-scaling
curl -X POST "http://localhost:8000/api/v1/auto-scaling/start-scaling" \
  -H "Authorization: Bearer YOUR_TOKEN"

# 5. Create alert rule
curl -X POST "http://localhost:8000/api/v1/intelligent-alerting/rules" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "high_cpu",
    "name": "High CPU Alert",
    "description": "Alert when CPU usage is high",
    "metric_name": "cpu_usage",
    "condition": "gt",
    "threshold": 80.0,
    "severity": "warning"
  }'

# 6. Start alerting
curl -X POST "http://localhost:8000/api/v1/intelligent-alerting/start-alerting" \
  -H "Authorization: Bearer YOUR_TOKEN"

# 7. Check system status
curl -X GET "http://localhost:8000/api/v1/advanced-analytics/summary" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

## Support

For API support and questions:

1. Check the troubleshooting section in the Advanced Features Guide
2. Review error messages and status codes
3. Consult the API documentation
4. Contact the development team

## Changelog

### Version 2.0.0
- Added Advanced Analytics API with ML capabilities
- Implemented Auto-Scaling API with predictive scaling
- Created Advanced Dashboard API with real-time monitoring
- Added Intelligent Alerting API with multiple notification channels
- Enhanced error handling and rate limiting
- Added comprehensive API documentation
