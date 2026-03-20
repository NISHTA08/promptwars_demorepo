import hashlib
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

_db = None

def _get_db():
    global _db
    if _db is not None:
        return _db
        
    import os
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        logger.warning("No GOOGLE_APPLICATION_CREDENTIALS set. Skipping Firestore to avoid hang.")
        return None

    try:
        import firebase_admin
        from firebase_admin import credentials, firestore
        
        # Initialize the app with default credentials if not already initialized
        if not firebase_admin._apps:
            firebase_admin.initialize_app()
        _db = firestore.client()
    except ImportError:
        logger.error("firebase-admin package is not installed.")
    except Exception as e:
        logger.error(f"Failed to initialize Firestore client: {e}")
        
    return _db

def log_signal(
    raw_input_text: str, 
    domain: str, 
    urgency: str, 
    pipeline_trace: Dict[str, Any], 
    token_counts: Dict[str, int]
) -> Optional[str]:
    """
    Logs the processed signal data to the 'signal_logs' Firestore collection.
    
    CRITICAL: For privacy, we NEVER log the raw_input_text.
    Instead, we log the SHA-256 hash of the input text.
    """
    db = _get_db()
    if not db:
        logger.warning("Firestore DB not available, skipping logging.")
        return None
        
    # Hash the text to avoid storing raw PII
    input_hash = hashlib.sha256(raw_input_text.encode('utf-8')).hexdigest()
    
    try:
        from firebase_admin import firestore
        log_data = {
            "input_hash": input_hash,
            "domain": domain,
            "urgency": urgency,
            "pipeline_trace": pipeline_trace,
            "token_counts": token_counts,
            "created_at": firestore.SERVER_TIMESTAMP
        }
        
        _, doc_ref = db.collection("signal_logs").add(log_data)
        logger.info(f"Successfully logged signal trace to Firestore. Document ID: {doc_ref.id}")
        return doc_ref.id
        
    except Exception as e:
        logger.error(f"Error writing to Firestore 'signal_logs' collection: {e}")
        return None
