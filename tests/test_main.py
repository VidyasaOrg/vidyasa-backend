"""Test cases for the main FastAPI application endpoints"""
import pytest


class TestMainApp:
    """Test cases for main application endpoints"""
    
    def test_root_endpoint(self, client):
        """Test the root endpoint returns welcome message"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "Welcome to Vidyasa Backend!" in data["message"]
    
    def test_health_check_endpoint(self, client):
        """Test the health check endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_nonexistent_endpoint_returns_404(self, client):
        """Test accessing non-existent endpoint returns 404"""
        response = client.get("/nonexistent")
        
        assert response.status_code == 404
