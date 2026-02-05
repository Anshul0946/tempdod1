import requests
import json
import time
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from typing import Dict, Optional, Any
from .config import API_BASE

class APIManager:
    """
    Handles all external API interactions with fault tolerance.
    """
    def __init__(self, token: str):
        self.token = token
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",  # Required by some providers
            "X-Title": "Cellular Processor",
        }

    def validate_api_key(self) -> bool:
        """Simple check to see if token looks valid."""
        if not self.token or len(self.token.strip()) < 20:
            return False
        return True

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), 
           retry=retry_if_exception_type((requests.exceptions.Timeout, requests.exceptions.ConnectionError)))
    def post_chat_completion(self, payload: Dict[str, Any], timeout: int = 60) -> Optional[Dict[str, Any]]:
        """
        Robust API call with retries.
        """
        url = f"{API_BASE}/chat/completions"
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            # Don't retry 4xx errors (client error), but let 5xx (server error) retry or raise
            if 400 <= e.response.status_code < 500:
                print(f"API Client Error: {e}")
                return None
            raise e
        except Exception as e:
             raise e # Allow tenacity to catch it

    def clean_json_response(self, content: str) -> str:
        """Helper to remove markdown from JSON responses."""
        if not content:
            return ""
        content = content.strip()
        if content.startswith("```"):
            import re
            content = re.sub(r'^```(?:json)?\s*\n?', '', content)
            content = re.sub(r'\n?```\s*$', '', content)
        return content.strip()
