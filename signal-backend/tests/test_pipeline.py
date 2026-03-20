"""
Tests for the 4-step agentic pipeline.
All Gemini calls are mocked with realistic JSON fixtures.
"""
import json
import pytest
from unittest.mock import patch, MagicMock

from pipeline.intake import parse_and_normalize
from pipeline.verify import verify
from pipeline.reason import reason
from pipeline.action import decide_action


# ── Realistic Gemini Fixtures per Domain ────────────────────

INTAKE_FIXTURES = {
    "medical": json.dumps({
        "raw_cleaned": "Patient reports severe chest pain radiating to left arm for 30 minutes. History of hypertension. Currently taking amlodipine 5mg daily.",
        "detected_language": "en",
        "input_type_confirmed": "text",
        "entities_raw": {
            "people": ["patient"],
            "locations": [],
            "times": ["30 minutes"],
            "quantities": ["5mg"],
            "medical_terms": ["chest pain", "hypertension", "amlodipine"],
            "legal_terms": [],
            "severity_signals": ["severe", "radiating to left arm"],
            "contact_info": []
        },
        "missing_critical": ["patient name", "age", "location"],
        "parse_confidence": 0.91
    }),
    "disaster": json.dumps({
        "raw_cleaned": "Flooding reported in sector 14 Gurgaon. Water level rising above 3 feet. Around 200 families stranded. Need immediate rescue boats.",
        "detected_language": "en",
        "input_type_confirmed": "text",
        "entities_raw": {
            "people": [],
            "locations": ["sector 14 Gurgaon"],
            "times": [],
            "quantities": ["3 feet", "200 families"],
            "medical_terms": [],
            "legal_terms": [],
            "severity_signals": ["stranded", "immediate rescue"],
            "contact_info": []
        },
        "missing_critical": ["reporter name", "contact number"],
        "parse_confidence": 0.88
    }),
    "legal": json.dumps({
        "raw_cleaned": "Tenant served eviction notice dated March 15 2026 by landlord Ramesh Kumar at 42 MG Road Bangalore. 30 day notice period cited under Karnataka Rent Control Act.",
        "detected_language": "en",
        "input_type_confirmed": "text",
        "entities_raw": {
            "people": ["Ramesh Kumar"],
            "locations": ["42 MG Road Bangalore"],
            "times": ["March 15 2026", "30 day notice period"],
            "quantities": [],
            "medical_terms": [],
            "legal_terms": ["eviction notice", "Karnataka Rent Control Act"],
            "severity_signals": [],
            "contact_info": []
        },
        "missing_critical": ["tenant name"],
        "parse_confidence": 0.94
    }),
    "civic": json.dumps({
        "raw_cleaned": "Large pothole on NH-48 near Manesar toll plaza causing accidents. Two vehicles damaged today. No barricades or warning signs present.",
        "detected_language": "en",
        "input_type_confirmed": "text",
        "entities_raw": {
            "people": [],
            "locations": ["NH-48", "Manesar toll plaza"],
            "times": ["today"],
            "quantities": ["two vehicles"],
            "medical_terms": [],
            "legal_terms": [],
            "severity_signals": ["causing accidents", "no barricades"],
            "contact_info": []
        },
        "missing_critical": ["reporter contact"],
        "parse_confidence": 0.87
    }),
}


# ── Tests for Step 1: Intake (parse_and_normalize) ──────────

class TestIntakePipeline:
    """Tests for pipeline/intake.py — Step 1: Parse & Normalize."""

    @pytest.mark.parametrize("domain", ["medical", "disaster", "legal", "civic"])
    @patch("pipeline.intake._call_gemini_with_retry")
    def test_intake_returns_valid_structure(self, mock_gemini, domain):
        """Each domain should return the expected JSON keys."""
        mock_gemini.return_value = INTAKE_FIXTURES[domain]

        result = parse_and_normalize("some raw input text for testing", domain_hint=domain)

        assert "raw_cleaned" in result
        assert "detected_language" in result
        assert "input_type_confirmed" in result
        assert "entities_raw" in result
        assert "missing_critical" in result
        assert "parse_confidence" in result
        mock_gemini.assert_called_once()

    @patch("pipeline.intake._call_gemini_with_retry")
    def test_intake_medical_entities(self, mock_gemini):
        """Medical domain should extract medical_terms and severity_signals."""
        mock_gemini.return_value = INTAKE_FIXTURES["medical"]

        result = parse_and_normalize("patient has severe chest pain", domain_hint="medical")
        entities = result["entities_raw"]

        assert "chest pain" in entities["medical_terms"]
        assert "hypertension" in entities["medical_terms"]
        assert len(entities["severity_signals"]) > 0

    @patch("pipeline.intake._call_gemini_with_retry")
    def test_intake_disaster_entities(self, mock_gemini):
        """Disaster domain should extract locations and quantities."""
        mock_gemini.return_value = INTAKE_FIXTURES["disaster"]

        result = parse_and_normalize("flooding in sector 14", domain_hint="disaster")
        entities = result["entities_raw"]

        assert "sector 14 Gurgaon" in entities["locations"]
        assert len(entities["quantities"]) > 0

    @patch("pipeline.intake._call_gemini_with_retry")
    def test_intake_legal_entities(self, mock_gemini):
        """Legal domain should extract legal_terms and people."""
        mock_gemini.return_value = INTAKE_FIXTURES["legal"]

        result = parse_and_normalize("eviction notice received", domain_hint="legal")
        entities = result["entities_raw"]

        assert "eviction notice" in entities["legal_terms"]
        assert "Ramesh Kumar" in entities["people"]

    @patch("pipeline.intake._call_gemini_with_retry")
    def test_intake_civic_entities(self, mock_gemini):
        """Civic domain should extract locations and severity_signals."""
        mock_gemini.return_value = INTAKE_FIXTURES["civic"]

        result = parse_and_normalize("pothole on highway", domain_hint="civic")
        entities = result["entities_raw"]

        assert "NH-48" in entities["locations"]
        assert "causing accidents" in entities["severity_signals"]

    @patch("pipeline.intake._call_gemini_with_retry")
    def test_intake_invalid_json_triggers_retry(self, mock_gemini):
        """If Gemini returns invalid JSON first, it should retry once."""
        mock_gemini.side_effect = [
            "This is not valid JSON at all",  # First call fails
            INTAKE_FIXTURES["medical"],        # Retry succeeds
        ]

        result = parse_and_normalize("chest pain report", domain_hint="medical")

        assert result["detected_language"] == "en"
        assert mock_gemini.call_count == 2

    @patch("pipeline.intake._call_gemini_with_retry")
    def test_intake_confidence_is_float(self, mock_gemini):
        """parse_confidence should always be a float."""
        mock_gemini.return_value = INTAKE_FIXTURES["disaster"]

        result = parse_and_normalize("flood report", domain_hint="disaster")

        assert isinstance(result["parse_confidence"], float)


# ── Tests for Step 2: Verify (stub) ─────────────────────────

class TestVerifyPipeline:
    """Tests for pipeline/verify.py — Step 2: Verify."""

    def test_verify_returns_dict(self):
        intake_data = json.loads(INTAKE_FIXTURES["medical"])
        result = verify(intake_data, domain="medical")
        assert isinstance(result, dict)

    def test_verify_passes_through_intake(self):
        intake_data = json.loads(INTAKE_FIXTURES["disaster"])
        result = verify(intake_data, domain="disaster")
        assert result["verified"] is True
        assert result["intake_result"] == intake_data


# ── Tests for Step 3: Reason (stub) ─────────────────────────

class TestReasonPipeline:
    """Tests for pipeline/reason.py — Step 3: Reason."""

    def test_reason_returns_urgency(self):
        verified_data = {"verified": True, "intake_result": {}}
        result = reason(verified_data, domain="legal")
        assert "urgency" in result

    def test_reason_returns_required_fields(self):
        verified_data = {"verified": True, "intake_result": {}}
        result = reason(verified_data, domain="civic")
        assert "recommended_actions" in result
        assert "reasoning_chain" in result
        assert "ambiguities" in result
        assert "confidence" in result
        assert "requires_human_review" in result


# ── Tests for Step 4: Action (stub) ─────────────────────────

class TestActionPipeline:
    """Tests for pipeline/action.py — Step 4: Action."""

    def test_action_returns_action_type(self):
        reason_data = {"urgency": "high", "recommended_actions": []}
        result = decide_action(reason_data, domain="disaster")
        assert "action_type" in result

    def test_action_returns_location(self):
        reason_data = {"urgency": "medium", "recommended_actions": []}
        result = decide_action(reason_data, domain="civic")
        assert "location_name" in result
