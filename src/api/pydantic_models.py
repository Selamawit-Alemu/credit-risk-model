# src/api/pydantic_models.py

from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):

    Amount: float = Field(..., example=150.0)
    PricingStrategy: float = Field(..., example=2)
    transaction_hour: int = Field(..., ge=0, le=23, example=13)
    transaction_day: int = Field(..., ge=1, le=31, example=14)
    transaction_month: int = Field(..., ge=1, le=12, example=7)
    transaction_year: int = Field(..., example=2024)
    total_amount: float = Field(..., example=1200.0)
    avg_amount: float = Field(..., example=300.0)
    count_transactions: float = Field(..., example=4)
    std_amount: float = Field(..., example=45.8)
    ProductCategory: str = Field(..., example="Electronics")
    ChannelId: str = Field(..., example="Mobile")
    ProviderId: str = Field(..., example="ProviderA")
class PredictionResponse(BaseModel):
    is_high_risk: int
    risk_probability: float
