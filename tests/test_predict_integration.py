import pytest
from fastapi.testclient import TestClient

from api.main import app  # Adjust if your FastAPI app is in a different module
from api.dependencies import get_predictor


class MockPredictor:
    """Mock ML predictor used for API integration testing."""
    def predict(self, df):
        return [250000.567]


@pytest.fixture
def client():
    """
    Creates a test client with the predictor dependency overridden.
    """
    def override_get_predictor():
        return MockPredictor()

    app.dependency_overrides[get_predictor] = override_get_predictor

    with TestClient(app) as test_client:
        yield test_client

    # Important: clear dependency overrides after the test
    app.dependency_overrides.clear()


class TestPredictAPI:

    def test_predict_price_success(self, client):
        """
        Integration test with Valid request to test API end-to-end
        """
        payload = {"longitude": -122.23, "latitude": 37.88,
            "housing_median_age": 41, "total_rooms": 880,
            "total_bedrooms": 129, "population": 322,
            "households": 126, "median_income": 8.3252,
            "ocean_proximity": "NEAR BAY"
        }
        response = client.post("/predict", json=payload)
        # Verify HTTP status
        assert response.status_code == 200
        data = response.json()
        # Verify response structure
        assert "predicted_median_house_value" in data
        assert "status_code" in data
        # Verify prediction
        assert data["predicted_median_house_value"] == 250000.57
        # Verify application status code
        assert data["status_code"] == 200

    def test_predict_price_invalid_request(self, client):
        """
        Integration test with invalid request (Missing a required field)
        """
        invalid_payload = {
            "longitude": -122.23, "latitude": 37.88,
            "housing_median_age": 41, # total_rooms is intentionally missing
            "total_bedrooms": 129, "population": 322,
            "households": 126, "median_income": 8.3252,
            "ocean_proximity": "NEAR BAY"
        }
        response = client.post("/predict", json=invalid_payload)
        assert response.status_code == 422

    def test_predict_price_invalid_ocean_proximity(self, client):
        """
        Test schema validation for invalid categorical input.
        """
        payload = {
            "longitude": -122.23, "latitude": 37.88,
            "housing_median_age": 41, "total_rooms": 880,
            "total_bedrooms": 129, "population": 322,
            "households": 126, "median_income": 8.3252,
            "ocean_proximity": True
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_price_with_empty_payload(self, client):
        """
        Test schema validation for invalid categorical input.
        """
        response = client.post("/predict", json={})
        assert response.status_code == 422
