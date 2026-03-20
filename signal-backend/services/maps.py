import os
import requests
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Simple in-memory cache to store geocoding results keyed by location string
_GEOCODE_CACHE: Dict[str, Dict[str, Any]] = {}

def geocode_location(location_string: str) -> Optional[Dict[str, Any]]:
    """
    Geocodes a location string into coordinates using the Google Maps Geocoding API.
    Results are cached to avoid redundant API calls.
    """
    if not location_string:
        return None
        
    normalized_location = location_string.strip().lower()
    
    # Check cache first
    if normalized_location in _GEOCODE_CACHE:
        logger.info(f"Returning cached geocode result for '{location_string}'")
        return _GEOCODE_CACHE[normalized_location]
        
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        logger.error("GOOGLE_MAPS_API_KEY environment variable is not set. Cannot geocode.")
        return None
        
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": location_string,
        "key": api_key
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") == "OK" and data.get("results"):
            result = data["results"][0]
            # Store safely in cache
            _GEOCODE_CACHE[normalized_location] = result
            return result
        else:
            logger.warning(f"Geocoding failed for '{location_string}'. API Status: {data.get('status')}")
            return None
            
    except requests.RequestException as e:
        logger.error(f"Error communicating with Google Maps API: {e}")
        return None
