from pydantic import BaseModel


class PredictionResponse(BaseModel):

    tumor_detected: bool

    tumor_type: str

    confidence: float