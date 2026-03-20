import json
import logging
from typing import Dict, Any

from services.gemini import _call_gemini_with_retry

logger = logging.getLogger(__name__)

INTAKE_SYSTEM_PROMPT = """You are the intake parser for Signal. You receive raw unstructured input.
Your ONLY job is extraction — no reasoning, no recommendations.

Extract strictly what is present. Use null for anything absent.

domain_hint: {domain_hint}

Return ONLY this JSON:
{{
  "raw_cleaned": "cleaned version of input, typos fixed, slang normalized",
  "detected_language": "ISO 639-1 code",
  "input_type_confirmed": "text|voice|image|document",
  "entities_raw": {{
    "people": ["string"],
    "locations": ["string"],
    "times": ["string"],
    "quantities": ["string"],
    "medical_terms": ["string"],
    "legal_terms": ["string"],
    "severity_signals": ["string"],
    "contact_info": ["string"]
  }},
  "missing_critical": ["string"],
  "parse_confidence": 0.0
}}

Rules: No hallucination. No inference. Only extract what is present."""

def parse_and_normalize(raw_text: str, domain_hint: str) -> Dict[str, Any]:
    """
    Step 1 of the agentic chain.
    Receives raw text and domain_hint, calls Gemini, and returns parsed entities.
    """
    system_instruction = INTAKE_SYSTEM_PROMPT.format(domain_hint=domain_hint)
    
    # Combine the system instructions with the raw input text
    prompt = f"{system_instruction}\n\nRaw Text to Parse:\n{raw_text}"
    
    # Call Gemini to get the structured JSON response
    response_text = _call_gemini_with_retry(prompt, max_attempts=3)
    
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        logger.warning("Intake parse failed: Invalid JSON. Retrying with explicit structure instruction.")
        retry_prompt = prompt + "\n\nReturn ONLY the JSON object, no other text."
        response_text = _call_gemini_with_retry(retry_prompt, max_attempts=1)
        data = json.loads(response_text)
        
    return data
