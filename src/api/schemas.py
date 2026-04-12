"""request and response schemas."""
from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    age: int = Field(..., ge=18, le=100)
    gender: str
    city: str
    tenure_months: int = Field(..., ge=0)
    tariff_id: str
    contract_type: str
    payment_method: str
    paperless_billing: int = Field(..., ge=0, le=1)
    has_internet: int = Field(..., ge=0, le=1)
    has_tv: int = Field(..., ge=0, le=1)
    has_roaming: int = Field(..., ge=0, le=1)
    recency: float = 30.0
    frequency: float = 6.0
    monetary: float = 5000.0
    avg_data_gb: float = 15.0
    avg_minutes: float = 300.0

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 34, "gender": "M", "city": "Almaty",
                "tenure_months": 24, "tariff_id": "T03",
                "contract_type": "monthly", "payment_method": "card",
                "paperless_billing": 1, "has_internet": 1,
                "has_tv": 0, "has_roaming": 0,
                "recency": 5, "frequency": 12,
                "monetary": 48000, "avg_data_gb": 22.5, "avg_minutes": 480.0,
            }
        }
    }


class ChurnResponse(BaseModel):
    churn_probability: float
    churn_predicted: bool
    risk_level: str
    top_factors: list[dict]


class RecommendRequest(BaseModel):
    customer_id: str | None = None
    current_tariff: str | None = None
    category: str | None = None
    mode: str = "cold_start"   # cold_start | similar | hybrid
    n: int = Field(5, ge=1, le=10)

    model_config = {
        "json_schema_extra": {
            "example": {
                "customer_id": "C000001",
                "current_tariff": "T03",
                "mode": "similar",
                "n": 5,
            }
        }
    }


class TariffRecommendation(BaseModel):
    tariff_id: str
    name: str
    monthly_price: int
    data_gb: int
    category: str
    score: float


class RecommendResponse(BaseModel):
    mode: str
    recommendations: list[TariffRecommendation]


class SegmentRequest(BaseModel):
    recency: float
    frequency: float
    monetary: float
    tenure_months: float
    avg_data_gb: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "recency": 5, "frequency": 12,
                "monetary": 48000, "tenure_months": 24, "avg_data_gb": 22.5,
            }
        }
    }


class SegmentResponse(BaseModel):
    segment_id: int
    segment_name: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict[str, bool]
