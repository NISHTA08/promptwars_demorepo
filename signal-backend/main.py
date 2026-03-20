import os
import re
import uuid
import logging
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from models.schemas import AnalyseRequest, AnalyseResponse, FeedbackRequest
from pipeline.intake import parse_and_normalize
from pipeline.verify import verify
from pipeline.reason import reason
from pipeline.action import decide_action
from services.firestore import log_signal

# ── Logging ──────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Rate Limiter ─────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── App ──────────────────────────────────────────────────
app = FastAPI(title="Signal Backend")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS ─────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Input Sanitization ───────────────────────────────────
def sanitize_text(text: str) -> str:
    """Remove potentially dangerous characters while preserving meaning."""
    text = text.strip()
    text = text.replace("\x00", "")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<[^>]*>", "", text)
    return text

# ── Serve Frontend ───────────────────────────────────────
# Path to signal-app.html (one directory up from signal-backend)
FRONTEND_PATH = Path(__file__).resolve().parent.parent / "signal-app.html"

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    """Serve the Signal frontend HTML page."""
    if FRONTEND_PATH.exists():
        return HTMLResponse(content=FRONTEND_PATH.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Frontend not found</h1>", status_code=404)

# ── Endpoints ────────────────────────────────────────────

@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "Signal Backend is running!"}


@app.post("/api/analyse", response_model=AnalyseResponse)
@limiter.limit("20/minute")
async def analyse(request: Request, body: AnalyseRequest):
    """
    Main analysis endpoint.
    Runs the 4-step agentic pipeline sequentially:
      1. Intake  → parse & normalize
      2. Verify  → cross-check extracted entities
      3. Reason  → determine urgency, actions, ambiguities
      4. Action  → decide final action type
    """
    request_id = str(uuid.uuid4())
    pipeline_trace: Dict[str, Any] = {"request_id": request_id}
    token_counts: Dict[str, int] = {}

    # Sanitize the input text
    clean_text = sanitize_text(body.text)

    # ── Step 1: Intake ───────────────────────────────────
    intake_result = parse_and_normalize(clean_text, domain_hint=body.domain)
    pipeline_trace["step_1_intake"] = intake_result

    # ── Step 2: Verify ───────────────────────────────────
    verify_result = verify(intake_result, domain=body.domain)
    pipeline_trace["step_2_verify"] = verify_result

    # ── Step 3: Reason ───────────────────────────────────
    reason_result = reason(verify_result, domain=body.domain)
    pipeline_trace["step_3_reason"] = reason_result

    # ── Step 4: Action ───────────────────────────────────
    action_result = decide_action(reason_result, domain=body.domain)
    pipeline_trace["step_4_action"] = action_result

    # ── Assemble response ────────────────────────────────
    # Entities as key-value dict (frontend renders key:value pairs)
    entities = intake_result.get("entities_raw", {})

    response = AnalyseResponse(
        domain=body.domain,
        domain_confidence=intake_result.get("parse_confidence", 0.0),
        urgency=reason_result.get("urgency", "MEDIUM"),
        urgency_reason=reason_result.get("urgency_reason", 
            f"Analysis based on {body.domain} domain signals."),
        entities=entities,
        recommended_actions=reason_result.get("recommended_actions", []),
        reasoning_chain=reason_result.get("reasoning_chain", []),
        ambiguities=reason_result.get("ambiguities", []),
        location_name=action_result.get("location_name"),
        confidence=reason_result.get("confidence", 0.0),
        requires_human_review=reason_result.get("requires_human_review", True),
        action_type=action_result.get("action_type", "alert"),
        pipeline_trace=pipeline_trace,
    )

    # ── Log to Firestore (non-blocking) ──────────────────
    try:
        log_signal(
            raw_input_text=clean_text,
            domain=body.domain,
            urgency=response.urgency,
            pipeline_trace=pipeline_trace,
            token_counts=token_counts,
        )
    except Exception as e:
        logger.error(f"Firestore logging failed: {e}")

    return response


@app.post("/api/feedback")
@limiter.limit("20/minute")
async def feedback(request: Request, body: FeedbackRequest):
    """Receives user feedback on a previous analysis."""
    logger.info(
        f"Feedback received — request_id={body.request_id}, "
        f"was_helpful={body.was_helpful}, correction={body.correction}"
    )
    return {"status": "received", "request_id": body.request_id}
