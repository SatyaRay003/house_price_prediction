"""
Unit tests Schema & Predict function.
"""

import pytest
from unittest.mock import Mock
from pydantic import ValidationError
from fastapi import HTTPException

from api.routers.predict import predict_price
from api.schemas import HouseFeaturesRequest

# Unit test of the schema

def test_invalid_ocean_proximity():
    with pytest.raises(ValidationError):
        HouseFeaturesRequest(
            longitude=-122.23,
            latitude=37.88,
            housing_median_age=41,
            total_rooms=1000,
            total_bedrooms=200,
            population=500,
            households=100,
            median_income=5.0,
            ocean_proximity=True # ocean_proximity is a string field but boolean value provided
        )

def test_missing_required_field():
    with pytest.raises(ValidationError):
        HouseFeaturesRequest(
            longitude=-122.23,
            latitude=37.88,
            housing_median_age=41,
            total_rooms=1000,
            # total_bedrooms missing
            population=500,
            households=100,
            median_income=5.0,
            ocean_proximity="INLAND"
        )

# Unit test of the predict API

@pytest.fixture
def valid_payload():
    return HouseFeaturesRequest(
        longitude=-122.23,
        latitude=37.88,
        housing_median_age=41,
        total_rooms=1000,
        total_bedrooms=200,
        population=500,
        households=100,
        median_income=5.0,
        ocean_proximity="INLAND"
    )

@pytest.fixture
def mock_predictor():
    predictor = Mock()
    predictor.predict.return_value = [250000.567]
    return predictor

def test_successful_prediction(valid_payload, mock_predictor):
    response = predict_price(payload=valid_payload, predictor=mock_predictor)

    assert response.predicted_median_house_value == 250000.57
    assert response.status_code == 200

    mock_predictor.predict.assert_called_once()

def test_prediction_failure(valid_payload):
    failing_predictor = Mock()
    failing_predictor.predict.side_effect = ValueError("Model prediction failed")

    with pytest.raises(HTTPException) as exc_info:
        predict_price(payload=valid_payload, predictor=failing_predictor)

    assert exc_info.value.status_code == 400
    assert "Inference calculation failed" in exc_info.value.detail


# def test_feature_engineering(valid_payload, mock_predictor):
#     predict_price(payload=valid_payload, predictor=mock_predictor)

#     passed_df = mock_predictor.predict.call_args[0][0]

#     assert passed_df["RoomsPerHousehold"].iloc[0] == 10
#     assert passed_df["BedroomsPerRoom"].iloc[0] == 0.2
#     assert passed_df["PopulationPerHousehold"].iloc[0] == 5

# def test_categorical_encoding(valid_payload, mock_predictor):
#     predict_price(payload=valid_payload, predictor=mock_predictor)

#     passed_df = mock_predictor.predict.call_args[0][0]

#     assert passed_df["ocean_proximity_INLAND"].iloc[0] == 1
#     assert passed_df["ocean_proximity_ISLAND"].iloc[0] == 0
#     assert passed_df["ocean_proximity_NEAR BAY"].iloc[0] == 0
#     assert passed_df["ocean_proximity_NEAR OCEAN"].iloc[0] == 0
