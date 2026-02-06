import base64
import json
import re
import requests
from pathlib import Path
from typing import Optional
from .config import (
    ProcessingContext, ServiceData, SpeedTestData, VideoTestData, VoiceCallData,
    SERVICE_SCHEMA_JSON, SPEED_SCHEMA_JSON, VIDEO_SCHEMA_JSON, VOICE_SCHEMA_JSON, 
    LLMProvider, OCR_URL
)
from .api_manager import APIManager

# ==================== REASONING PROMPTS (OCR-AWARE) ====================

# Tell reasoning model about common OCR mistakes
OCR_CONTEXT = """
IMPORTANT: This text comes from OCR and may contain errors:
- 'l' or 'I' misread as '1' (or vice versa)
- 'O' misread as '0' (or vice versa)  
- Extra quotes or symbols like '104 instead of 104
- Units attached to numbers like "363Mbps" or "45ms"
- Spaces in wrong places

Use common sense to fix these OCR errors when extracting values.
"""

PROMPT_SERVICE = f"""Parse this OCR text from cellular network info screens into JSON.
{OCR_CONTEXT}

Output this exact JSON structure:
{{
    "nr_arfcn": <number>,
    "nr_band": <number>,
    "nr_pci": <number>,
    "nr_bw": <number>,
    "nr5g_rsrp": <number, typically negative like -85>,
    "nr5g_rsrq": <number, typically negative>,
    "nr5g_sinr": <number>,
    "lte_band": <number>,
    "lte_earfcn": <number>,
    "lte_pci": <number>,
    "lte_bw": <number>,
    "lte_rsrp": <number, typically negative>,
    "lte_rsrq": <number, typically negative>,
    "lte_sinr": <number>
}}

Look for: NR_BAND, NR_BW, RSRP, RSRQ, SINR, LTE_BAND, EARFCN, PCI values.
Return ONLY the JSON with numeric values."""

PROMPT_SPEED = f"""Parse this speed test OCR text into JSON.
{OCR_CONTEXT}

Output this exact JSON structure:
{{
    "download_mbps": <number>,
    "upload_mbps": <number>,
    "ping_ms": <number>,
    "jitter_ms": <number>
}}

Look for Download/Upload speeds in Mbps, Ping/Latency in ms, Jitter in ms.
Fix any OCR errors and extract clean numeric values only.
Return ONLY the valid JSON object."""

PROMPT_VIDEO = f"""Parse this video streaming test OCR text into JSON.
{OCR_CONTEXT}

Output this exact JSON structure:
{{
    "max_resolution": "<string like 1080p, 4K, 720p>",
    "load_time_ms": <number>,
    "buffering_percentage": <number 0-100>
}}

Return ONLY the valid JSON object."""

PROMPT_VOICE = f"""Parse this voice call screen data into JSON.
{OCR_CONTEXT}

Output this exact JSON structure:
{{
    "phone_number": "<string with phone number>",
    "call_duration_seconds": <number in seconds>,
    "call_status": "<string: Connected, Completed, Failed, etc>",
    "time": "<string with call time>"
}}

Convert duration to seconds (e.g., "1:30" = 90, "0l:30" = 90).
Return ONLY the valid JSON object."""


class Extractor:
    """
    Structured Pipelines:
    - Service: OCR + OCR → Reasoning
    - Speed:   OCR → Reasoning (with number cleaning)
    - Video:   OCR → Reasoning (with number cleaning)
    - Voice:   Vision → Reasoning (uses vision model for call screens)
    """
    
    def __init__(self, api_manager: APIManager, context: ProcessingContext):
        self.api = api_manager
        self.context = context

    def _encode_image(self, path: str) -> Optional[str]:
        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            self.context.log(f"[ERROR] Cannot read {path}: {e}")
            return None

    def _clean_number(self, value):
        """Clean a value to be a valid number."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return value
        # Remove quotes, units, etc.
        s = str(value).strip().strip("'\"")
        # Extract number from string like "363 Mbps" or "-85 dBm"
        match = re.search(r'(-?\d+\.?\d*)', s)
        if match:
            try:
                return float(match.group(1))
            except:
                return None
        return None

    # ==================== PADDLEOCR ====================
    def _paddle_ocr(self, image_path: str, api_key: str) -> Optional[str]:
        """PaddleOCR for text extraction."""
        name = Path(image_path).stem
        self.context.log(f"[OCR] {name}")
        
        b64 = self._encode_image(image_path)
        if not b64:
            return None
        
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
            
            texts = []
            if "data" in data:
                for item in data["data"]:
                    if "text_detections" in item:
                        for detection in item["text_detections"]:
                            if "text_prediction" in detection:
                                text = detection["text_prediction"].get("text", "")
                                if text:
                                    texts.append(text)
            
            if texts:
                ocr_text = "\n".join(texts)
                self.context.log(f"[OCR] Got {len(ocr_text)} chars ({len(texts)} lines)")
                return ocr_text
            else:
                self.context.log(f"[OCR] No text found")
                return None
                
        except Exception as e:
            self.context.log(f"[OCR ERROR] {e}")
            return None

    # ==================== VISION MODEL (for Voice) ====================
    def _vision_extract(self, image_path: str, prompt: str, vision_provider: LLMProvider) -> Optional[str]:
        """Vision model for complex images like call screens."""
        name = Path(image_path).stem
        self.context.log(f"[VISION] {name}")
        
        b64 = self._encode_image(image_path)
        if not b64:
            return None

        payload = {
            "model": vision_provider.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
        }

        try:
            resp = self.api.post_chat_completion(payload, provider=vision_provider)
            if resp and "choices" in resp:
                text = resp["choices"][0]["message"]["content"]
                self.context.log(f"[VISION] Got {len(text)} chars")
                return text
        except Exception as e:
            self.context.log(f"[VISION ERROR] {e}")
        return None

    # ==================== REASONING PARSER ====================
    def _reasoning_parse(self, text: str, prompt: str, schema_cls, 
                          reasoning_provider: LLMProvider, clean_numbers: bool = False):
        """Parse text using reasoning model."""
        self.context.log(f"[PARSE] Reasoning...")
        
        full_prompt = f"{prompt}\n\nTEXT TO PARSE:\n{text}"
        
        try:
            json_str = self.api.post_reasoning_completion(full_prompt, reasoning_provider)
            if json_str:
                data = json.loads(json_str)
                
                # Clean numbers if needed (for Speed/Video)
                if clean_numbers:
                    for key, value in data.items():
                        if key not in ['max_resolution', 'call_status', 'phone_number', 'time']:
                            data[key] = self._clean_number(value)
                
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

    # ==================== SERVICE PIPELINE ====================
    def analyze_service_images(self, sector: str, img1: str, img2: str,
                                vision_provider: LLMProvider,
                                reasoning_provider: LLMProvider) -> Optional[ServiceData]:
        """Service: OCR both images → Reasoning merges."""
        self.context.log(f"--- {sector.upper()} SERVICE ---")
        
        t1 = self._paddle_ocr(img1, vision_provider.api_key)
        t2 = self._paddle_ocr(img2, vision_provider.api_key)
        
        if not t1 and not t2:
            self.context.log(f"[ERROR] No OCR text for {sector}")
            return None
        
        combined = f"IMAGE 1:\n{t1 or 'empty'}\n\nIMAGE 2:\n{t2 or 'empty'}"
        return self._reasoning_parse(combined, PROMPT_SERVICE, ServiceData, reasoning_provider)

    # ==================== SPEED PIPELINE ====================
    def analyze_speed_test(self, img: str, vision_provider: LLMProvider,
                           reasoning_provider: LLMProvider) -> Optional[SpeedTestData]:
        """Speed: OCR → Reasoning with number cleaning."""
        name = Path(img).stem
        self.context.log(f"--- SPEED: {name} ---")
        
        text = self._paddle_ocr(img, vision_provider.api_key)
        if not text:
            return None
        
        return self._reasoning_parse(text, PROMPT_SPEED, SpeedTestData, 
                                     reasoning_provider, clean_numbers=True)

    # ==================== VIDEO PIPELINE ====================
    def analyze_video_test(self, img: str, vision_provider: LLMProvider,
                           reasoning_provider: LLMProvider) -> Optional[VideoTestData]:
        """Video: OCR → Reasoning with number cleaning."""
        name = Path(img).stem
        self.context.log(f"--- VIDEO: {name} ---")
        
        text = self._paddle_ocr(img, vision_provider.api_key)
        if not text:
            return None
        
        return self._reasoning_parse(text, PROMPT_VIDEO, VideoTestData, 
                                     reasoning_provider, clean_numbers=True)

    # ==================== VOICE PIPELINE (Uses Vision Model) ====================
    def analyze_voice_call(self, img: str, vision_provider: LLMProvider,
                           reasoning_provider: LLMProvider) -> Optional[VoiceCallData]:
        """Voice: Vision Model → Reasoning (better for call screens)."""
        name = Path(img).stem
        self.context.log(f"--- VOICE: {name} ---")
        
        # Use Vision model to read call screen
        vision_prompt = """Read this phone call screenshot. Extract:
- The phone number being called
- Call duration (if shown)
- Call status (Connected, Completed, Failed, Ringing, etc.)
- Time of the call

Return all the information you can see."""
        
        text = self._vision_extract(img, vision_prompt, vision_provider)
        if not text:
            return None
        
        return self._reasoning_parse(text, PROMPT_VOICE, VoiceCallData, 
                                     reasoning_provider, clean_numbers=True)
