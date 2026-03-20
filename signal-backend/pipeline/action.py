# Action pipeline logic — Step 4 of the agentic chain
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def decide_action(reason_result: Dict[str, Any], domain: str) -> Dict[str, Any]:
    """
    Step 4: Decide the final action based on reasoning output.
    TODO: Implement Gemini action-decision logic.
    """
    logger.info("Step 4 (action): placeholder — returning default action.")
    return {
        "action_type": f"{domain}_response",
        "location_name": None,
    }
