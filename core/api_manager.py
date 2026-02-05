import requests
import json
import time
import re
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

    def post_chat_completion(self, payload: Dict[str, Any], provider = None) -> Optional[Dict[str, Any]]:
        """
        Robust API call with manual double-timeout logic.
        Attempt 1: 60s timeout.
        Attempt 2: 120s timeout.
        
        provider: Optional[LLMProvider] - If provided, uses this provider's URL and Key.
                  Otherwise uses the default token and API_BASE.
        """
        
        # Determine URL and Headers based on provider override
        if provider:
            url = f"{provider.base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {provider.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "Cellular Processor",
            }
        else:
            url = f"{API_BASE}/chat/completions"
            headers = self.headers

        timeouts = [60, 120]
        
        for i, timeout in enumerate(timeouts):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=timeout)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as e:
                # 4xx errors are likely client errors, do not retry unless 408/429?
                # For simplicity, if it's 400-499, we fail immediately (except 429)
                if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                    print(f"[API] Client Error {e.response.status_code}: {e}")
                    raise e # Let caller handle logic error
                
                print(f"[API] HTTP Error {e} (Attempt {i+1})")
                if i < len(timeouts) - 1:
                    time.sleep(2)
                    continue
                raise e
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                print(f"[API] Network Error: {e} (Attempt {i+1})")
                if i < len(timeouts) - 1:
                    print(f"[API] Retrying with extended timeout ({timeouts[i+1]}s)...")
                    time.sleep(2)
                    continue
                raise e # Final failure
            except Exception as e:
                raise e # Unexpected error

        return None

    def clean_json_response(self, content: str) -> str:
        """Helper to remove markdown from JSON responses."""
        if not content:
            return ""
        
        # 1. Try finding the first outer brace pair
        # This regex looks for { followed by anything (non-greedy) until the last }
        match = re.search(r'(\{.*\})', content, re.DOTALL)
        if match:
             content = match.group(1)
        
        content = content.strip()
        return content
