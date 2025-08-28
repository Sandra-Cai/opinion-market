import pytest
from fastapi.testclient import TestClient
from app.main_simple import app

client = TestClient(app)

def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "Opinion Market API" in data["message"]
    assert "features" in data

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "opinion-market-api"

def test_readiness_check():
    """Test readiness check endpoint"""
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["service"] == "opinion-market-api"

def test_metrics_endpoint():
    """Test metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert "requests_total" in data["metrics"]

def test_api_root():
    """Test API root endpoint"""
    response = client.get("/api/v1/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "endpoints" in data

def test_api_health():
    """Test API health endpoint"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["api_version"] == "v1"

def test_get_markets():
    """Test get markets endpoint"""
    response = client.get("/api/v1/markets")
    assert response.status_code == 200
    data = response.json()
    assert "markets" in data
    assert len(data["markets"]) > 0
    assert "id" in data["markets"][0]
    assert "title" in data["markets"][0]

def test_docs_endpoint():
    """Test API documentation endpoint"""
    response = client.get("/docs")
    assert response.status_code == 200

def test_openapi_endpoint():
    """Test OpenAPI schema endpoint"""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert "openapi" in data

def test_basic_functionality():
    """Test basic functionality"""
    assert 2 + 2 == 4
    assert "hello" + " world" == "hello world"
    assert len("test") == 4
