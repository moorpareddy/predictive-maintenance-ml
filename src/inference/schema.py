from pydantic import BaseModel, Field
from typing import List


class EngineFeatures(BaseModel):
    features: List[float] = Field(
        ..., description="Ordered list of engineered sensor features"
    )


class PredictionResponse(BaseModel):
    will_fail_soon: int
    probability: float
