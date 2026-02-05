import time
from pathlib import Path
from typing import List, Dict, Any
from .config import ProcessingContext, LLMProvider
from .api_manager import APIManager
from .file_handler import FileHandler
from .extractor import Extractor
from .mapper import Mapper

class Evaluator:
    """
    Workflow orchestrator with deterministic routing.
    Now supports dual-model pipeline (Vision + Reasoning).
    """
    def __init__(self, context: ProcessingContext, api: APIManager, file_handler: FileHandler, extractor: Extractor, mapper: Mapper):
        self.context = context
        self.api = api
        self.file_handler = file_handler
        self.extractor = extractor
        self.mapper = mapper

    def _get_image_for_sector(self, sector: str, suffix: str) -> str:
        candidate = f"{sector}_{suffix}"
        return self.file_handler.get_image_path(candidate)

    def verify_completeness(self, reasoning_provider: LLMProvider):
        """
        Rule 2: Check if data is complete. 
        Uses reasoning model for validation if needed.
        """
        self.context.log("[EVAL] Rule 2: Verifying data completeness...")

        # Check Alpha Service
        if not self.context.alpha_service.nr_arfcn:
            self.context.log("[EVAL] Alpha Service missing fields.")
        
        # Check Beta Service
        if not self.context.beta_service.nr_arfcn:
            self.context.log("[EVAL] Beta Service missing fields.")
            
        # Check Gamma Service
        if not self.context.gamma_service.nr_arfcn:
            self.context.log("[EVAL] Gamma Service missing fields.")

    def process_workflow(
        self, 
        xlsx_path: str, 
        provider_vision: LLMProvider,
        provider_reasoning: LLMProvider
    ) -> str:
        """
        Main orchestration pipeline with Dual-Model approach:
        - Vision provider: extracts raw text from images
        - Reasoning provider: parses text into structured JSON
        """
        self.context.log("=" * 50)
        self.context.log("STARTING DUAL-MODEL PIPELINE")
        self.context.log(f"Vision: {provider_vision.model}")
        self.context.log(f"Reasoning: {provider_reasoning.model}")
        self.context.log("=" * 50)
        
        # 1. Extract Images
        images = self.file_handler.extract_images_from_excel(xlsx_path)
        if not images:
            self.context.log("[ERROR] No images found.")
            return None

        # 2. Process Service Images (Alpha, Beta, Gamma) - Images 1 & 2
        for sector in ["alpha", "beta", "gamma"]:
            img1 = self.file_handler.get_image_path(f"{sector}_image_1")
            img2 = self.file_handler.get_image_path(f"{sector}_image_2")
            
            if img1 and img2:
                self.context.log(f"\n--- Processing {sector.upper()} SERVICE ---")
                data = self.extractor.analyze_service_images(
                    sector, img1, img2, 
                    provider_vision, 
                    provider_reasoning
                )
                if data:
                    setattr(self.context, f"{sector}_service", data)
                    self.context.log(f"[SUCCESS] {sector} service data stored.")
                
                time.sleep(1)  # Delay between sectors to avoid rate limits

        # 3. Process Generic Images (Speed, Video, Voice) with Deterministic Routing
        self.context.log("\n--- Processing GENERIC IMAGES ---")
        for img_path in images:
            name = Path(img_path).stem
            parts = name.split("_")
            
            # FIX: Edge case where split produces fewer parts than expected
            if len(parts) >= 3 and parts[1] == "image":
                sector = parts[0]
                try:
                    suffix = int(parts[2])
                except ValueError:
                    self.context.log(f"[WARN] Could not parse number from {name}")
                    continue 

                # ----------------- ROUTING LOGIC -----------------
                # Voice Pipeline
                if sector == "voicetest":
                    data = self.extractor.analyze_voice_call(img_path, provider_vision, provider_reasoning)
                    if data:
                        self.context.voice_test[name] = data
                    time.sleep(0.5)
                    continue

                # Service Pipeline (Already handled above)
                if suffix == 1 or suffix == 2:
                    continue

                # Speed Pipeline (Images 3-7)
                if 3 <= suffix <= 7:
                    data = self.extractor.analyze_speed_test(img_path, provider_vision, provider_reasoning)
                    if data:
                        getattr(self.context, f"{sector}_speedtest")[name] = data
                    time.sleep(0.5)
                    continue

                # Video Pipeline (Image 8)
                if suffix == 8:
                    data = self.extractor.analyze_video_test(img_path, provider_vision, provider_reasoning)
                    if data:
                        getattr(self.context, f"{sector}_video")[name] = data
                    time.sleep(0.5)
                    continue
                
                # Unknown/Extra Images
                self.context.log(f"[WARN] Image {name} did not match any pipeline (Suffix {suffix}). Skipping.")

        # 4. Rule 2 Verify
        self.context.log("\n--- VERIFICATION PHASE ---")
        self.verify_completeness(provider_reasoning)

        # 5. Map to Excel
        self.context.log("\n--- MAPPING PHASE ---")
        self.mapper.map_to_excel(xlsx_path)
        
        self.context.log("=" * 50)
        self.context.log("PIPELINE COMPLETE")
        self.context.log("=" * 50)
        
        return xlsx_path
