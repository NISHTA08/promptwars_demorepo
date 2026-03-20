import os
import json
import time
import logging

logger = logging.getLogger(__name__)

# Lazy-loaded Gemini model reference
_model = None

def _get_model():
    """Lazily initialize the Gemini model so imports don't crash without an API key."""
    global _model
    if _model is None:
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", ""))
        _model = genai.GenerativeModel('gemini-1.5-pro')
    return _model

def _call_gemini_with_retry(prompt: str, max_attempts: int = 3) -> str:
    """Calls Gemini API with exponential backoff on failure."""
    import google.generativeai as genai
    import os

    if not os.environ.get("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY is not set. Cannot call Gemini API.")

    model = _get_model()

    generation_config = genai.GenerationConfig(
        temperature=0.2,
        response_mime_type="application/json"
    )

    for attempt in range(max_attempts):
        try:
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            return response.text
        except Exception as e:
            if attempt == max_attempts - 1:
                logger.error(f"Gemini API call failed after {max_attempts} attempts: {e}")
                raise

            sleep_time = 2 ** attempt
            logger.warning(f"Error from Gemini API: {e}. Retrying in {sleep_time}s (Attempt {attempt + 1}/{max_attempts})...")
            time.sleep(sleep_time)

    raise RuntimeError("Failed to generate content from Gemini API after retries.")

def analyze_signal(text: str, domain: str, input_type: str) -> dict:
    """
    Sends the request to Gemini and parses the response.
    Falls back and appends explicit JSON instructions if decoding fails.
    """
    from models.schemas import AnalyseResponse

    prompt = (
        f"Please analyze the following text.\n"
        f"Domain: {domain}\n"
        f"Input Type: {input_type}\n\n"
        f"Text to analyze:\n{text}\n"
    )

    response_text = _call_gemini_with_retry(prompt, max_attempts=3)

    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        logger.warning("Invalid JSON response received. Retrying with explicit JSON instruction.")
        retry_prompt = prompt + "\nReturn ONLY the JSON object, no other text."
        response_text = _call_gemini_with_retry(retry_prompt, max_attempts=1)
        data = json.loads(response_text)

    return data
