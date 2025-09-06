import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestAPIEndpoints:
    """Test API endpoints"""

    def test_api_root(self):
        """Test API root endpoint"""
        response = client.get("/api/v1/")
        assert response.status_code == 200

    def test_api_health(self):
        """Test API health endpoint"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_api_metrics(self):
        """Test API metrics endpoint"""
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200

    def test_api_docs(self):
        """Test API documentation"""
        response = client.get("/api/v1/docs")
        assert response.status_code == 200

    def test_api_openapi(self):
        """Test OpenAPI schema"""
        response = client.get("/api/v1/openapi.json")
        assert response.status_code == 200
        assert "openapi" in response.json()
