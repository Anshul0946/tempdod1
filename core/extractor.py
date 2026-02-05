import base64
import json
from pathlib import Path
from typing import Optional, Dict, Any
from .config import (
    ProcessingContext, ServiceData, SpeedTestData, VideoTestData, VoiceCallData,
    SERVICE_SCHEMA_JSON, SPEED_SCHEMA_JSON, VIDEO_SCHEMA_JSON, VOICE_SCHEMA_JSON, LLMProvider
)
from .api_manager import APIManager

class Extractor:
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

    def analyze_service_images(self, sector: str, img1_path: str, img2_path: str, provider: LLMProvider) -> Optional[ServiceData]:
        self.context.log(f"[AI] Analyzing Service Data for {sector} using {provider.name}...")
        b1 = self._encode_image(img1_path)
        b2 = self._encode_image(img2_path)
        if not b1 or not b2:
            return None

        prompt = (
            "You are a hyper-specialized AI for cellular network engineering data analysis. "
            "Analyze both provided service-mode screenshots carefully and return exactly one JSON object "
            "matching the schema. Use null where value is not found.\n\n"
            f"SCHEMA:\n{json.dumps(SERVICE_SCHEMA_JSON, indent=2)}"
        )

        payload = {
            "model": provider.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b1}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b2}"}},
                    ],
                }
            ],
            "response_format": {"type": "json_object"},
        }

        try:
            resp = self.api.post_chat_completion(payload, provider=provider)
            if resp:
                content = self.api.clean_json_response(resp["choices"][0]["message"]["content"])
                try:
                    data_dict = json.loads(content)
                    return ServiceData(**data_dict)
                except Exception as e:
                    self.context.log(f"[ERROR] Failed to parse Service Data for {sector}: {e}")
                    self.context.log(f"[DEBUG] Raw Content: {content}")
            else:
                self.context.log(f"[ERROR] API Request failed for {sector} service data.")
        except Exception as e:
            self.context.log(f"[ERROR] Exception during Service Analysis for {sector}: {e}")
            
        return None

    def analyze_speed_test(self, image_path: str, provider: LLMProvider) -> Optional[SpeedTestData]:
        name = Path(image_path).stem
        self.context.log(f"[AI] Analyzing Speed Test Data: {name} using {provider.name}")
        b = self._encode_image(image_path)
        if not b: return None

        prompt = (
            "Extract Speed Test data from this screenshot. Return strict JSON matching the schema.\n"
            f"SCHEMA:\n{json.dumps(SPEED_SCHEMA_JSON, indent=2)}"
        )
        return self._run_single_image_analysis(name, b, prompt, provider, SpeedTestData)

    def analyze_video_test(self, image_path: str, provider: LLMProvider) -> Optional[VideoTestData]:
        name = Path(image_path).stem
        self.context.log(f"[AI] Analyzing Video Test Data: {name} using {provider.name}")
        b = self._encode_image(image_path)
        if not b: return None

        prompt = (
            "Extract Video Test data from this screenshot. Return strict JSON matching the schema.\n"
            f"SCHEMA:\n{json.dumps(VIDEO_SCHEMA_JSON, indent=2)}"
        )
        return self._run_single_image_analysis(name, b, prompt, provider, VideoTestData)

    def analyze_voice_call(self, image_path: str, provider: LLMProvider) -> Optional[VoiceCallData]:
        name = Path(image_path).stem
        self.context.log(f"[AI] Analyzing Voice Call Data: {name} using {provider.name}")
        b = self._encode_image(image_path)
        if not b: return None

        prompt = (
            "Extract Voice Call data from this screenshot. Return strict JSON matching the schema.\n"
            f"SCHEMA:\n{json.dumps(VOICE_SCHEMA_JSON, indent=2)}"
        )
        return self._run_single_image_analysis(name, b, prompt, provider, VoiceCallData)

    def _run_single_image_analysis(self, name: str, b64_img: str, prompt: str, provider: LLMProvider, schema_cls):
        payload = {
            "model": provider.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}},
                    ],
                }
            ],
            "response_format": {"type": "json_object"},
        }
        
        try:
            resp = self.api.post_chat_completion(payload, provider=provider)
            if resp:
                content = self.api.clean_json_response(resp["choices"][0]["message"]["content"])
                try:
                    data = json.loads(content)
                    return schema_cls(**data)
                except Exception as e:
                    self.context.log(f"[ERROR] JSON Parse Error for {name}: {e}")
                    self.context.log(f"[DEBUG] Raw Content: {content}")
            else:
                 self.context.log(f"[ERROR] API Request failed for {name}")
        except Exception as e:
            self.context.log(f"[ERROR] Exception during analysis of {name}: {e}")
            
        return None
