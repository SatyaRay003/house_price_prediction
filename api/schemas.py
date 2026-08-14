"""
Request and Response Data Models
"""
from pydantic import BaseModel, Field

class HouseFeaturesRequest(BaseModel):
    longitude: float = Field(..., description="Longitude coordinate", json_schema_extra={"example": -122.23})
    latitude: float = Field(..., description="Latitude coordinate", json_schema_extra={"example": 37.88})
    housing_median_age: float = Field(..., ge=0, description="Median age of house", json_schema_extra={"example": 41.0})
    total_rooms: float = Field(..., gt=0, description="Total rooms in block", json_schema_extra={"example": 880.0})
    total_bedrooms: float = Field(..., gt=0, description="Total bedrooms in block", json_schema_extra={"example": 129.0})
    population: float = Field(..., gt=0, description="Population in block", json_schema_extra={"example": 322.0})
    households: float = Field(..., gt=0, description="Households in block", json_schema_extra={"example": 126.0})
    median_income: float = Field(..., ge=0, description="Median income (tens of thousands USD)", json_schema_extra={"example": 8.3252})
    ocean_proximity: str = Field(..., description="Location category", json_schema_extra={"example": "NEAR BAY"})

class PredictionResponse(BaseModel):
    predicted_median_house_value: float
    status_code: int
    model_version: str = "RandomForest_v2"
