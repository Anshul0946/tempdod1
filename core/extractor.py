import base64
import json
from pathlib import Path
from typing import Optional, Dict, Any
from .config import (
    ProcessingContext, ServiceData, SpeedTestData, VideoTestData, VoiceCallData,
    SERVICE_SCHEMA_JSON, GENERIC_SCHEMAS
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

    def analyze_service_images(self, sector: str, img1_path: str, img2_path: str, model_name: str) -> Optional[ServiceData]:
        self.context.log(f"[AI] Analyzing Service Data for {sector}...")
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
            "model": model_name,
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

        resp = self.api.post_chat_completion(payload)
        if resp:
            try:
                content = self.api.clean_json_response(resp["choices"][0]["message"]["content"])
                data_dict = json.loads(content)
                # Pydantic Validation
                return ServiceData(**data_dict)
            except Exception as e:
                self.context.log(f"[ERROR] Failed to parse Service Data for {sector}: {e}")
        return None

    def analyze_generic_image(self, image_path: str, model_name: str) -> Dict[str, Any]:
        """
        Returns a dict with 'image_type' and 'data'.
        """
        name = Path(image_path).stem
        self.context.log(f"[AI] Analyzing Generic Image: {name}")
        b = self._encode_image(image_path)
        if not b:
             return {}

        prompt = (
            "You are an expert AI assistant for analyzing cellular network test data. "
            "Classify the image as 'speed_test', 'video_test', or 'voice_call' and return a single JSON object "
            "matching the corresponding schema. Use null for missing fields.\n\n"
            f"SCHEMAS:\n{json.dumps(GENERIC_SCHEMAS, indent=2)}"
        )

        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b}"}},
                    ],
                }
            ],
            "response_format": {"type": "json_object"},
        }

        resp = self.api.post_chat_completion(payload)
        if resp:
            try:
                content = self.api.clean_json_response(resp["choices"][0]["message"]["content"])
                result = json.loads(content)
                img_type = result.get("image_type")
                raw_data = result.get("data", {})
                
                # Validation
                if img_type == "speed_test":
                    validated = SpeedTestData(**raw_data)
                    return {"image_type": "speed_test", "data": validated}
                elif img_type == "video_test":
                    validated = VideoTestData(**raw_data)
                    return {"image_type": "video_test", "data": validated}
                elif img_type == "voice_call":
                    validated = VoiceCallData(**raw_data)
                    return {"image_type": "voice_call", "data": validated}
                else:
                    self.context.log(f"[WARN] Unknown image type '{img_type}' for {name}")
                    return {}
            except Exception as e:
                self.context.log(f"[ERROR] Failed to parse Generic Data for {name}: {e}")
        return {}
