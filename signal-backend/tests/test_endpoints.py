"""
Tests for FastAPI endpoints: /api/analyse, /api/health, /api/feedback.
All Gemini calls are mocked. Tests cover all 4 domains, validation,
rate limiting, and feedback.
"""
import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


# ── Realistic Gemini Intake Fixtures ────────────────────────

INTAKE_FIXTURES = {
    "medical": {
        "raw_cleaned": "Patient reports severe chest pain radiating to left arm for 30 minutes. History of hypertension.",
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
        "missing_critical": ["patient name", "age"],
        "parse_confidence": 0.91
    },
    "disaster": {
        "raw_cleaned": "Flooding reported in sector 14. Water level rising above 3 feet. Around 200 families stranded.",
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
        "missing_critical": ["reporter contact"],
        "parse_confidence": 0.88
    },
    "legal": {
        "raw_cleaned": "Tenant served eviction notice dated March 15 by landlord Ramesh Kumar at 42 MG Road.",
        "detected_language": "en",
        "input_type_confirmed": "text",
        "entities_raw": {
            "people": ["Ramesh Kumar"],
            "locations": ["42 MG Road Bangalore"],
            "times": ["March 15 2026", "30 day notice"],
            "quantities": [],
            "medical_terms": [],
            "legal_terms": ["eviction notice", "Rent Control Act"],
            "severity_signals": [],
            "contact_info": []
        },
        "missing_critical": ["tenant name"],
        "parse_confidence": 0.94
    },
    "civic": {
        "raw_cleaned": "Large pothole on NH-48 near Manesar toll plaza causing accidents. Two vehicles damaged today.",
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
    },
}


# ── Health Endpoint ─────────────────────────────────────────

class TestHealthEndpoint:

    def test_health_returns_ok(self):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "message" in data


# ── Analyse Endpoint — All 4 Domains ───────────────────────

class TestAnalyseEndpoint:

    @pytest.mark.parametrize("domain", ["medical", "disaster", "legal", "civic"])
    @patch("main.log_signal", return_value=None)
    @patch("main.parse_and_normalize")
    def test_analyse_returns_200_for_each_domain(self, mock_intake, mock_log, domain):
        """POST /api/analyse should return 200 with valid structure for each domain."""
        mock_intake.return_value = INTAKE_FIXTURES[domain]

        payload = {
            "text": "This is a sufficiently long test input for the analyse endpoint.",
            "domain": domain,
            "input_type": "text"
        }
        resp = client.post("/api/analyse", json=payload)

        assert resp.status_code == 200
        body = resp.json()
        assert body["domain"] == domain
        assert "urgency" in body
        assert "entities" in body
        assert "recommended_actions" in body
        assert "reasoning_chain" in body
        assert "confidence" in body
        assert "action_type" in body
        assert "pipeline_trace" in body

    @patch("main.log_signal", return_value=None)
    @patch("main.parse_and_normalize")
    def test_analyse_medical_extracts_entities(self, mock_intake, mock_log):
        """Medical domain should return medical_terms in the entities list."""
        mock_intake.return_value = INTAKE_FIXTURES["medical"]

        payload = {
            "text": "Patient reports severe chest pain radiating to left arm for 30 minutes.",
            "domain": "medical",
            "input_type": "text"
        }
        resp = client.post("/api/analyse", json=payload)

        assert resp.status_code == 200
        body = resp.json()
        assert "chest pain" in body["entities"]["medical_terms"]
        assert "hypertension" in body["entities"]["medical_terms"]

    @patch("main.log_signal", return_value=None)
    @patch("main.parse_and_normalize")
    def test_analyse_disaster_extracts_locations(self, mock_intake, mock_log):
        """Disaster domain should include extracted locations in entities."""
        mock_intake.return_value = INTAKE_FIXTURES["disaster"]

        payload = {
            "text": "Flooding reported in sector 14 Gurgaon. Water level rising above 3 feet.",
            "domain": "disaster",
            "input_type": "text"
        }
        resp = client.post("/api/analyse", json=payload)

        assert resp.status_code == 200
        body = resp.json()
        assert "sector 14 Gurgaon" in body["entities"]["locations"]

    @patch("main.log_signal", return_value=None)
    @patch("main.parse_and_normalize")
    def test_analyse_pipeline_trace_present(self, mock_intake, mock_log):
        """pipeline_trace should contain all 4 steps and a request_id."""
        mock_intake.return_value = INTAKE_FIXTURES["civic"]

        payload = {
            "text": "Large pothole on NH-48 near Manesar toll plaza causing two accidents.",
            "domain": "civic",
            "input_type": "text"
        }
        resp = client.post("/api/analyse", json=payload)

        body = resp.json()
        trace = body["pipeline_trace"]
        assert "request_id" in trace
        assert "step_1_intake" in trace
        assert "step_2_verify" in trace
        assert "step_3_reason" in trace
        assert "step_4_action" in trace


# ── Input Validation ────────────────────────────────────────

class TestInputValidation:

    def test_empty_input_rejected(self):
        """Text shorter than 10 characters should be rejected (422)."""
        payload = {
            "text": "",
            "domain": "medical",
            "input_type": "text"
        }
        resp = client.post("/api/analyse", json=payload)
        assert resp.status_code == 422

    def test_short_input_rejected(self):
        """Text with fewer than 10 characters should be rejected."""
        payload = {
            "text": "too short",
            "domain": "medical",
            "input_type": "text"
        }
        resp = client.post("/api/analyse", json=payload)
        assert resp.status_code == 422

    def test_oversized_input_rejected(self):
        """Text exceeding 4000 characters should be rejected (422)."""
        payload = {
            "text": "x" * 4001,
            "domain": "medical",
            "input_type": "text"
        }
        resp = client.post("/api/analyse", json=payload)
        assert resp.status_code == 422

    def test_missing_domain_rejected(self):
        """Missing required 'domain' field should be rejected."""
        payload = {
            "text": "A sufficiently long input text for validation testing.",
            "input_type": "text"
        }
        resp = client.post("/api/analyse", json=payload)
        assert resp.status_code == 422

    def test_missing_text_rejected(self):
        """Missing required 'text' field should be rejected."""
        payload = {
            "domain": "medical",
            "input_type": "text"
        }
        resp = client.post("/api/analyse", json=payload)
        assert resp.status_code == 422

    def test_max_length_input_accepted(self):
        """Exactly 4000 characters should be accepted."""
        with patch("main.parse_and_normalize", return_value=INTAKE_FIXTURES["medical"]), \
             patch("main.log_signal", return_value=None):
            payload = {
                "text": "x" * 4000,
                "domain": "medical",
                "input_type": "text"
            }
            resp = client.post("/api/analyse", json=payload)
            assert resp.status_code == 200


# ── Rate Limiting ───────────────────────────────────────────

class TestRateLimiting:

    @patch("main.log_signal", return_value=None)
    @patch("main.parse_and_normalize")
    def test_rate_limit_exceeded(self, mock_intake, mock_log):
        """Exceeding 20 requests/minute should return 429."""
        mock_intake.return_value = INTAKE_FIXTURES["medical"]

        payload = {
            "text": "This is a rate limit test input that is long enough to pass validation.",
            "domain": "medical",
            "input_type": "text"
        }

        # Reset the limiter state for a clean test
        app.state.limiter.reset()

        got_429 = False
        for i in range(25):
            resp = client.post("/api/analyse", json=payload)
            if resp.status_code == 429:
                got_429 = True
                break

        assert got_429, "Expected 429 rate limit response but never received one"


# ── Feedback Endpoint ───────────────────────────────────────

class TestFeedbackEndpoint:

    def test_feedback_returns_received(self):
        """POST /api/feedback should accept valid feedback."""
        payload = {
            "request_id": "abc-123-def-456",
            "was_helpful": True,
            "correction": None
        }
        resp = client.post("/api/feedback", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "received"
        assert body["request_id"] == "abc-123-def-456"

    def test_feedback_with_correction(self):
        """Feedback with a correction string should be accepted."""
        payload = {
            "request_id": "abc-123-def-456",
            "was_helpful": False,
            "correction": "The urgency should have been CRITICAL, not MEDIUM."
        }
        resp = client.post("/api/feedback", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "received"

    def test_feedback_missing_request_id(self):
        """Missing request_id should be rejected (422)."""
        payload = {
            "was_helpful": True
        }
        resp = client.post("/api/feedback", json=payload)
        assert resp.status_code == 422

    def test_feedback_missing_was_helpful(self):
        """Missing was_helpful should be rejected (422)."""
        payload = {
            "request_id": "abc-123"
        }
        resp = client.post("/api/feedback", json=payload)
        assert resp.status_code == 422
