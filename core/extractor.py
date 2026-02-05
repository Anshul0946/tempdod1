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
# These prompts are crafted for telecommunications expertise

PROMPT_VISION_SERVICE = """You are an expert telecommunications network engineer.
Read ALL text visible in this cellular service mode screenshot.
Focus on extracting these network parameters:
- 5G NR: ARFCN, Band, PCI, Bandwidth, RSRP, RSRQ, SINR
- 4G LTE: EARFCN, Band, PCI, Bandwidth, RSRP, RSRQ, SINR
Return the raw text exactly as displayed. Do NOT interpret values."""

PROMPT_VISION_SPEED = """You are a network performance analyst.
Read ALL text from this speed test screenshot.
Focus on: Download speed (Mbps), Upload speed (Mbps), Ping/Latency (ms), Jitter (ms).
Return raw text exactly as displayed."""

PROMPT_VISION_VIDEO = """You are a streaming quality analyst.
Read ALL text from this video streaming test screenshot.
Focus on: Maximum resolution, Load time, Buffering percentage.
Return raw text exactly as displayed."""

PROMPT_VISION_VOICE = """You are a voice quality analyst.
Read ALL text from this VoLTE/voice call screenshot.
Focus on: Phone number, Call duration, Call status, Time of call.
Return raw text exactly as displayed."""

PROMPT_REASONING_SERVICE = """You are an expert 5G/LTE cellular network engineer with deep knowledge of:
- 5G NR parameters: ARFCN (frequency), Band (n77/n78/etc), PCI (cell ID), Bandwidth, RSRP/RSRQ/SINR (signal quality)
- 4G LTE parameters: EARFCN, Band, PCI, Bandwidth, RSRP/RSRQ/SINR

Parse the following raw text into JSON matching the schema EXACTLY.
Rules:
- Extract numeric values only (no units like 'dBm' or 'MHz')
- RSRP is typically negative (-50 to -140 dBm)
- SINR can be positive or negative
- Use null ONLY if value is truly not present
- Do NOT hallucinate or guess values

Return ONLY the JSON object, no explanation."""

PROMPT_REASONING_SPEED = """You are a network performance expert.
Parse this speed test data into JSON matching the schema EXACTLY.
Rules:
- Download/Upload in Mbps (numeric only)
- Ping/Jitter in ms (numeric only)
- Use null if not found
Return ONLY the JSON object."""

PROMPT_REASONING_VIDEO = """You are a streaming quality expert.
Parse this video test data into JSON matching the schema EXACTLY.
Rules:
- Resolution as string (e.g., "1080p", "4K")
- Load time in ms
- Buffering as percentage (0-100)
Return ONLY the JSON object."""

PROMPT_REASONING_VOICE = """You are a voice call quality expert.
Parse this call data into JSON matching the schema EXACTLY.
Rules:
- Phone number as string with formatting
- Duration in seconds (convert if needed)
- Status as string (Connected, Failed, etc.)
- Time as string in original format
Return ONLY the JSON object."""


class Extractor:
    """
    Two-Stage Extraction Pipeline with API Call Optimization:
    1. Try local parsing first (no API call)
    2. If fails, use Vision → Reasoning pipeline
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

    def _try_local_parse(self, json_str: str, schema_cls):
        """
        Try to parse JSON locally without API call.
        Returns schema object if successful, None if needs reasoning API.
        
        STRICT: Any null value triggers full Reasoning pipeline.
        """
        try:
            if not json_str or json_str == "{}":
                return None
            data = json.loads(json_str)
            
            # STRICT CHECK: If ANY value is null, trigger reasoning
            has_null = any(v is None for v in data.values())
            if has_null:
                return None  # Force reasoning API call
            
            return schema_cls(**data)
        except:
            return None

    # ==================== STAGE 1: RAW TEXT EXTRACTION ====================
    def _extract_raw_text(self, image_path: str, vision_provider: LLMProvider, prompt: str) -> Optional[str]:
        """Vision model reads text from image with domain-specific prompt."""
        name = Path(image_path).stem
        self.context.log(f"[VISION] Extracting text from {name}...")
        
        b64_img = self._encode_image(image_path)
        if not b64_img:
            return None

        payload = {
            "model": vision_provider.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}},
                    ],
                }
            ],
        }

        try:
            resp = self.api.post_chat_completion(payload, provider=vision_provider)
            if resp and "choices" in resp:
                raw_text = resp["choices"][0]["message"]["content"]
                self.context.log(f"[VISION] Got {len(raw_text)} chars from {name}")
                return raw_text
        except Exception as e:
            self.context.log(f"[VISION ERROR] {name}: {e}")
        
        return None

    # ==================== STAGE 2: REASONING PARSE ====================
    def _parse_with_reasoning(self, raw_text: str, schema_json: dict, schema_cls, reasoning_provider: LLMProvider, prompt_template: str):
        """Reasoning model parses raw text into structured JSON."""
        self.context.log(f"[REASONING] Parsing extracted text...")
        
        prompt = f"{prompt_template}\n\nSCHEMA:\n{json.dumps(schema_json, indent=2)}\n\nRAW TEXT:\n{raw_text}"
        
        try:
            json_str = self.api.post_reasoning_completion(prompt, reasoning_provider)
            if json_str:
                data = json.loads(json_str)
                return schema_cls(**data)
        except Exception as e:
            self.context.log(f"[REASONING ERROR] Parse failed: {e}")
        
        return None

    def _smart_extract(self, image_path: str, vision_provider: LLMProvider, reasoning_provider: LLMProvider,
                       vision_prompt: str, reasoning_prompt: str, schema_json: dict, schema_cls):
        """
        Smart extraction with retry logic:
        1. Vision extracts with schema-aware prompt
        2. Try local parse
        3. If nulls found → Re-run Vision as pure OCR → Reasoning parses
        """
        name = Path(image_path).stem
        
        # Stage 1: Vision extraction with schema-aware prompt
        raw_text = self._extract_raw_text(image_path, vision_provider, vision_prompt)
        if not raw_text:
            return None
        
        # Try local parse
        local_json = self.api.clean_json_response(raw_text)
        local_result = self._try_local_parse(local_json, schema_cls)
        
        if local_result:
            self.context.log(f"[SUCCESS] {name}: All values extracted, no nulls")
            return local_result
        
        # ===== RETRY: Vision found nulls, re-extract as pure OCR =====
        self.context.log(f"[RETRY] {name}: Nulls detected, re-analyzing with OCR mode...")
        
        # Pure OCR prompt - just read all text, no schema
        ocr_prompt = (
            "You are an OCR assistant. Read and extract ALL text visible in this image. "
            "Return every piece of text you can see, including numbers, labels, and values. "
            "Do not interpret or structure the data, just read everything."
        )
        
        ocr_text = self._extract_raw_text(image_path, vision_provider, ocr_prompt)
        if not ocr_text:
            self.context.log(f"[WARN] {name}: OCR retry also failed")
            return None
        
        # Stage 2: Reasoning parses the OCR text
        self.context.log(f"[REASONING] {name}: Parsing OCR text with reasoning model...")
        return self._parse_with_reasoning(ocr_text, schema_json, schema_cls, reasoning_provider, reasoning_prompt)

    # ==================== SERVICE ANALYSIS (2 IMAGES → 1 SCHEMA) ====================
    def analyze_service_images(
        self, 
        sector: str, 
        img1_path: str, 
        img2_path: str, 
        vision_provider: LLMProvider,
        reasoning_provider: LLMProvider
    ) -> Optional[ServiceData]:
        """Service analysis: extract from both images, merge with reasoning."""
        self.context.log(f"[SERVICE] Processing {sector} (2 images)...")
        
        # Stage 1: Extract from both images
        text1 = self._extract_raw_text(img1_path, vision_provider, PROMPT_VISION_SERVICE)
        time.sleep(0.5)
        text2 = self._extract_raw_text(img2_path, vision_provider, PROMPT_VISION_SERVICE)
        
        if not text1 and not text2:
            self.context.log(f"[SERVICE ERROR] No text extracted for {sector}")
            return None
        
        # Combine and parse with reasoning (service needs merging, can't skip)
        combined = f"=== IMAGE 1 (Network Info) ===\n{text1 or 'No data'}\n\n=== IMAGE 2 (Signal Quality) ===\n{text2 or 'No data'}"
        
        result = self._parse_with_reasoning(combined, SERVICE_SCHEMA_JSON, ServiceData, reasoning_provider, PROMPT_REASONING_SERVICE)
        
        if result:
            self.context.log(f"[SERVICE OK] {sector} extracted successfully")
        
        return result

    # ==================== SPEED TEST ====================
    def analyze_speed_test(self, image_path: str, vision_provider: LLMProvider, reasoning_provider: LLMProvider) -> Optional[SpeedTestData]:
        return self._smart_extract(
            image_path, vision_provider, reasoning_provider,
            PROMPT_VISION_SPEED, PROMPT_REASONING_SPEED,
            SPEED_SCHEMA_JSON, SpeedTestData
        )

    # ==================== VIDEO TEST ====================
    def analyze_video_test(self, image_path: str, vision_provider: LLMProvider, reasoning_provider: LLMProvider) -> Optional[VideoTestData]:
        return self._smart_extract(
            image_path, vision_provider, reasoning_provider,
            PROMPT_VISION_VIDEO, PROMPT_REASONING_VIDEO,
            VIDEO_SCHEMA_JSON, VideoTestData
        )

    # ==================== VOICE CALL ====================
    def analyze_voice_call(self, image_path: str, vision_provider: LLMProvider, reasoning_provider: LLMProvider) -> Optional[VoiceCallData]:
        return self._smart_extract(
            image_path, vision_provider, reasoning_provider,
            PROMPT_VISION_VOICE, PROMPT_REASONING_VOICE,
            VOICE_SCHEMA_JSON, VoiceCallData
        )
