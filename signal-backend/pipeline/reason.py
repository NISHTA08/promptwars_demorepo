# Reason pipeline logic — Step 3 of the agentic chain
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def reason(verified_result: Dict[str, Any], domain: str) -> Dict[str, Any]:
    """
    Step 3: Reason over verified entities to produce urgency, actions, etc.
    TODO: Implement Gemini reasoning logic.
    """
    logger.info("Step 3 (reason): placeholder — returning default reasoning.")
    return {
        "urgency": "MEDIUM",
        "urgency_reason": f"Analysis based on {domain} domain signals.",
        "recommended_actions": [
            "Gather more information to classify the situation",
            "Identify the primary stakeholders involved",
            "Escalate to relevant authority",
        ],
        "reasoning_chain": [
            f"Domain detection: identified {domain} as primary domain.",
            "Entity extraction: structured entities from raw input.",
            "Contradiction check: no internal contradictions found.",
            "Urgency scoring: scored against domain rubric.",
        ],
        "ambiguities": ["Some details may be missing from the input"],
        "confidence": 0.72,
        "requires_human_review": True,
    }
