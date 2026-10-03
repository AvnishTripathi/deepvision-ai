"""
Tests for DeepVision AI Health and Documentation Endpoints.
"""

import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.mark.anyio
async def test_health_check_endpoint():
    """Verify GET /api/v1/health returns 200 OK and expected JSON schema."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data == {
            "status": "healthy",
            "service": "deepvision-api",
            "version": "0.1.0"
        }


@pytest.mark.anyio
async def test_root_endpoint():
    """Verify root endpoint provides service status and docs reference."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "deepvision-api"
        assert data["status"] == "online"
        assert "documentation" in data


@pytest.mark.anyio
async def test_openapi_documentation_accessible():
    """Verify standard OpenAPI documentation endpoints are available."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        docs_resp = await client.get("/docs")
        assert docs_resp.status_code == 200

        schema_resp = await client.get("/openapi.json")
        assert schema_resp.status_code == 200
        schema = schema_resp.json()
        assert "/api/v1/health" in schema["paths"]
