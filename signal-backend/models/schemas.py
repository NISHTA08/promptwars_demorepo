from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

class AnalyseRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=4000)
    domain: str
    input_type: str = "text"

class AnalyseResponse(BaseModel):
    domain: str
    domain_confidence: float = 0.0
    urgency: str
    urgency_reason: str = ""
    entities: Dict[str, Any]
    recommended_actions: List[str]
    reasoning_chain: List[str]
    ambiguities: List[str]
    location_name: Optional[str] = None
    confidence: float
    requires_human_review: bool
    action_type: str
    pipeline_trace: Dict[str, Any] = {}

class FeedbackRequest(BaseModel):
    request_id: str
    was_helpful: bool
    correction: Optional[str] = None
