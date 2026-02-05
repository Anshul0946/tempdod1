import base64
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any
from .config import (
    ProcessingContext, ServiceData, SpeedTestData, VideoTestData, VoiceCallData,
    SERVICE_SCHEMA_JSON, SPEED_SCHEMA_JSON, VIDEO_SCHEMA_JSON, VOICE_SCHEMA_JSON, LLMProvider
)
from .api_manager import APIManager

# ==================== DOMAIN-SPECIFIC PROMPTS ====================

PROMPT_OCR = """You are an expert OCR assistant. Read and extract ALL text visible in this image.
Return every piece of text you can see, including numbers, labels, values, and any UI elements.
Be thorough - do not miss any text. Just read and return all text, nothing else."""

PROMPT_REASONING_SERVICE = """You are an expert 5G/LTE cellular network engineer.
Parse the following raw text into JSON matching the schema EXACTLY.

RULES:
- Extract numeric values only (no units like 'dBm' or 'MHz')
- RSRP is typically negative (-50 to -140 dBm)
- SINR can be positive or negative
- If a value is not clearly present, use 0 instead of null
- Do NOT hallucinate - only extract values that are clearly visible in the text

Return ONLY the JSON object, no explanation."""

PROMPT_REASONING_SPEED = """You are a network performance expert.
Parse this speed test data into JSON matching the schema EXACTLY.

RULES:
- Download/Upload values should be in Mbps (numeric only)
- Ping/Latency should be in ms (numeric only)
- Jitter should be in ms (numeric only)
- If a value appears as text like "363 Mbps", extract just the number 363
- If a value is not found, use 0 instead of null

Return ONLY the JSON object, no other text."""

PROMPT_REASONING_VIDEO = """You are a streaming quality expert.
Parse this video test data into JSON matching the schema EXACTLY.

RULES:
- Resolution as string (e.g., "1080p", "4K", "720p")
- Load time in ms (numeric)
- Buffering as percentage 0-100 (numeric)
- If a value is not found, use 0 for numbers or "unknown" for strings

Return ONLY the JSON object, no other text."""

PROMPT_REASONING_VOICE = """You are a voice call quality expert.
Parse this call data into JSON matching the schema EXACTLY.

RULES:
- Phone number as string with formatting
- Duration: convert to seconds if needed (e.g., "1:30" = 90)
- Status as string
- Time as string in original format
- If a value is not found, use 0 for numbers or "unknown" for strings

Return ONLY the JSON object, no other text."""


class Extractor:
    """
    Simplified Two-Stage Pipeline:
    Stage 1: Vision (OCR) - extracts raw text
    Stage 2: Reasoning - parses into schema with NO NULLS
    """
    def __init__(self, api_manager: APIManager, context: ProcessingContext):
        self.api = api_manager
        self.context = context

    def _encode_image(self, path: str) -> Optional[str]:
        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            self.context.log(f"[ERROR] Could not read image {path}: {e}")
            return None

    def _extract_raw_text(self, image_path: str, vision_provider: LLMProvider) -> Optional[str]:
        """Stage 1: Vision extracts ALL text from image."""
        name = Path(image_path).stem
        self.context.log(f"[OCR] Reading {name}...")
        
        b64_img = self._encode_image(image_path)
        if not b64_img:
            return None

        payload = {
            "model": vision_provider.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT_OCR},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}},
                ],
            }],
        }

        try:
            resp = self.api.post_chat_completion(payload, provider=vision_provider)
            if resp and "choices" in resp:
                raw_text = resp["choices"][0]["message"]["content"]
                self.context.log(f"[OCR] Got {len(raw_text)} chars")
                return raw_text
        except Exception as e:
            self.context.log(f"[OCR ERROR] {name}: {e}")
        
        return None

    def _parse_with_reasoning(self, raw_text: str, schema_json: dict, schema_cls, 
                               reasoning_provider: LLMProvider, reasoning_prompt: str):
        """Stage 2: Reasoning parses text into schema."""
        self.context.log(f"[PARSE] Reasoning model parsing...")
        
        prompt = f"{reasoning_prompt}\n\nSCHEMA:\n{json.dumps(schema_json, indent=2)}\n\nRAW TEXT:\n{raw_text}"
        
        try:
            json_str = self.api.post_reasoning_completion(prompt, reasoning_provider)
            if json_str:
                data = json.loads(json_str)
                result = schema_cls(**data)
                self.context.log(f"[PARSE] Success")
                return result
                
        except Exception as e:
            self.context.log(f"[PARSE ERROR] {e}")
        
        return None

    # ==================== SERVICE ANALYSIS ====================
    def analyze_service_images(self, sector: str, img1_path: str, img2_path: str, 
                                vision_provider: LLMProvider, reasoning_provider: LLMProvider) -> Optional[ServiceData]:
        """Service: OCR both images → Reasoning merges."""
        self.context.log(f"[SERVICE] {sector.upper()} - 2 images")
        
        text1 = self._extract_raw_text(img1_path, vision_provider)
        time.sleep(0.3)
        text2 = self._extract_raw_text(img2_path, vision_provider)
        
        if not text1 and not text2:
            self.context.log(f"[SERVICE ERROR] No text for {sector}")
            return None
        
        combined = f"=== IMAGE 1 ===\n{text1 or 'No data'}\n\n=== IMAGE 2 ===\n{text2 or 'No data'}"
        result = self._parse_with_reasoning(combined, SERVICE_SCHEMA_JSON, ServiceData, 
                                            reasoning_provider, PROMPT_REASONING_SERVICE)
        
        if result:
            self.context.log(f"[SERVICE OK] {sector}")
        return result

    # ==================== SPEED TEST ====================
    def analyze_speed_test(self, image_path: str, vision_provider: LLMProvider, 
                           reasoning_provider: LLMProvider) -> Optional[SpeedTestData]:
        """Speed: OCR → Reasoning."""
        name = Path(image_path).stem
        
        raw_text = self._extract_raw_text(image_path, vision_provider)
        if not raw_text:
            return None
        
        result = self._parse_with_reasoning(raw_text, SPEED_SCHEMA_JSON, SpeedTestData,
                                            reasoning_provider, PROMPT_REASONING_SPEED)
        if result:
            self.context.log(f"[SPEED OK] {name}")
        return result

    # ==================== VIDEO TEST ====================
    def analyze_video_test(self, image_path: str, vision_provider: LLMProvider,
                           reasoning_provider: LLMProvider) -> Optional[VideoTestData]:
        """Video: OCR → Reasoning."""
        name = Path(image_path).stem
        
        raw_text = self._extract_raw_text(image_path, vision_provider)
        if not raw_text:
            return None
        
        result = self._parse_with_reasoning(raw_text, VIDEO_SCHEMA_JSON, VideoTestData,
                                            reasoning_provider, PROMPT_REASONING_VIDEO)
        if result:
            self.context.log(f"[VIDEO OK] {name}")
        return result

    # ==================== VOICE CALL ====================
    def analyze_voice_call(self, image_path: str, vision_provider: LLMProvider,
                           reasoning_provider: LLMProvider) -> Optional[VoiceCallData]:
        """Voice: OCR → Reasoning."""
        name = Path(image_path).stem
        
        raw_text = self._extract_raw_text(image_path, vision_provider)
        if not raw_text:
            return None
        
        result = self._parse_with_reasoning(raw_text, VOICE_SCHEMA_JSON, VoiceCallData,
                                            reasoning_provider, PROMPT_REASONING_VOICE)
        if result:
            self.context.log(f"[VOICE OK] {name}")
        return result
