import pytest
from fastapi.testclient import TestClient
from app.main_simple import app

@pytest.fixture
def client():
    """Create a test client for the simple app"""
    return TestClient(app)

@pytest.fixture
def app_instance():
    """Return the app instance"""
    return app
