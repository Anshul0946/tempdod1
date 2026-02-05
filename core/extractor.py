import base64
import json
import requests
from pathlib import Path
from typing import Optional
from .config import (
    ProcessingContext, ServiceData, SpeedTestData, VideoTestData, VoiceCallData,
    SERVICE_SCHEMA_JSON, SPEED_SCHEMA_JSON, VIDEO_SCHEMA_JSON, VOICE_SCHEMA_JSON, 
    LLMProvider, OCR_URL
)
from .api_manager import APIManager

# ==================== REASONING PROMPTS (Strict JSON) ====================

PROMPT_SERVICE = """Parse this OCR text from a cellular network info screen into JSON:

{
    "nr_arfcn": <number>,
    "nr_band": <number>,
    "nr_pci": <number>,
    "nr_bw": <number>,
    "nr5g_rsrp": <number, negative like -85>,
    "nr5g_rsrq": <number, negative>,
    "nr5g_sinr": <number>,
    "lte_band": <number>,
    "lte_earfcn": <number>,
    "lte_pci": <number>,
    "lte_bw": <number>,
    "lte_rsrp": <number, negative>,
    "lte_rsrq": <number, negative>,
    "lte_sinr": <number>
}

Extract numeric values only. Return ONLY the JSON."""

PROMPT_SPEED = """Parse this OCR text from a speed test screen into JSON:

{
    "download_mbps": <number>,
    "upload_mbps": <number>,
    "ping_ms": <number>,
    "jitter_ms": <number>
}

Look for: Download/Upload speeds in Mbps, Ping/Latency in ms, Jitter in ms.
Return ONLY the JSON."""

PROMPT_VIDEO = """Parse this OCR text from a video test screen into JSON:

{
    "max_resolution": "<string like 1080p or 4K>",
    "load_time_ms": <number>,
    "buffering_percentage": <number>
}

Return ONLY the JSON."""

PROMPT_VOICE = """Parse this OCR text from a voice call screen into JSON:

{
    "phone_number": "<string>",
    "call_duration_seconds": <number>,
    "call_status": "<string>",
    "time": "<string>"
}

If duration is "1:30", convert to 90 seconds.
Return ONLY the JSON."""


class Extractor:
    """
    Strict PaddleOCR + Reasoning Pipeline.
    No fallbacks, no retries - just OCR → Parse.
    """
    
    def __init__(self, api_manager: APIManager, context: ProcessingContext):
        self.api = api_manager
        self.context = context

    def _encode_image(self, path: str) -> Optional[str]:
        """Encode image to base64."""
        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            self.context.log(f"[ERROR] Cannot read {path}: {e}")
            return None

    def _paddle_ocr(self, image_path: str, api_key: str) -> Optional[str]:
        """
        Call PaddleOCR API to extract text from image.
        Returns raw OCR text.
        """
        name = Path(image_path).stem
        self.context.log(f"[OCR] {name}")
        
        b64 = self._encode_image(image_path)
        if not b64:
            return None
        
        # Check size limit
        if len(b64) >= 180000:
            self.context.log(f"[WARN] {name} too large for OCR")
            return None
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json"
        }
        
        payload = {
            "input": [{
                "type": "image_url",
                "url": f"data:image/png;base64,{b64}"
            }]
        }
        
        try:
            resp = requests.post(OCR_URL, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            
            # Extract text from PaddleOCR response
            # Response format: {"data": [{"text": [...]}]}
            if "data" in data and data["data"]:
                texts = []
                for item in data["data"]:
                    if isinstance(item, dict) and "text" in item:
                        texts.extend(item["text"])
                    elif isinstance(item, list):
                        texts.extend(item)
                
                ocr_text = " ".join(str(t) for t in texts if t)
                self.context.log(f"[OCR] Got {len(ocr_text)} chars")
                return ocr_text
            else:
                self.context.log(f"[OCR] No text found in response")
                return None
                
        except Exception as e:
            self.context.log(f"[OCR ERROR] {e}")
            return None

    def _reasoning_parse(self, text: str, prompt: str, schema_cls, 
                          reasoning_provider: LLMProvider):
        """Parse OCR text using reasoning model."""
        self.context.log(f"[PARSE] Reasoning...")
        
        full_prompt = f"{prompt}\n\nOCR TEXT:\n{text}"
        
        try:
            json_str = self.api.post_reasoning_completion(full_prompt, reasoning_provider)
            if json_str:
                data = json.loads(json_str)
                result = schema_cls(**data)
                
                # Log nulls
                nulls = [k for k, v in data.items() if v is None]
                if nulls:
                    self.context.log(f"[WARN] Nulls: {nulls}")
                else:
                    self.context.log(f"[OK] All fields parsed")
                return result
        except Exception as e:
            self.context.log(f"[PARSE ERROR] {e}")
        return None

    # ==================== SERVICE (2 images) ====================
    def analyze_service_images(self, sector: str, img1: str, img2: str,
                                vision_provider: LLMProvider,  # Contains API key
                                reasoning_provider: LLMProvider) -> Optional[ServiceData]:
        self.context.log(f"--- {sector.upper()} SERVICE ---")
        
        t1 = self._paddle_ocr(img1, vision_provider.api_key)
        t2 = self._paddle_ocr(img2, vision_provider.api_key)
        
        if not t1 and not t2:
            self.context.log(f"[ERROR] No OCR text for {sector}")
            return None
        
        combined = f"IMAGE 1: {t1 or 'empty'}\nIMAGE 2: {t2 or 'empty'}"
        return self._reasoning_parse(combined, PROMPT_SERVICE, ServiceData, reasoning_provider)

    # ==================== SPEED ====================
    def analyze_speed_test(self, img: str, vision_provider: LLMProvider,
                           reasoning_provider: LLMProvider) -> Optional[SpeedTestData]:
        text = self._paddle_ocr(img, vision_provider.api_key)
        if not text:
            return None
        return self._reasoning_parse(text, PROMPT_SPEED, SpeedTestData, reasoning_provider)

    # ==================== VIDEO ====================
    def analyze_video_test(self, img: str, vision_provider: LLMProvider,
                           reasoning_provider: LLMProvider) -> Optional[VideoTestData]:
        text = self._paddle_ocr(img, vision_provider.api_key)
        if not text:
            return None
        return self._reasoning_parse(text, PROMPT_VIDEO, VideoTestData, reasoning_provider)

    # ==================== VOICE ====================
    def analyze_voice_call(self, img: str, vision_provider: LLMProvider,
                           reasoning_provider: LLMProvider) -> Optional[VoiceCallData]:
        text = self._paddle_ocr(img, vision_provider.api_key)
        if not text:
            return None
        return self._reasoning_parse(text, PROMPT_VOICE, VoiceCallData, reasoning_provider)
