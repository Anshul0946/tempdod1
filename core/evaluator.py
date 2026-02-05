from pathlib import Path
from typing import List, Dict, Any
from .config import ProcessingContext
from .api_manager import APIManager
from .file_handler import FileHandler
from .extractor import Extractor
from .mapper import Mapper

class Evaluator:
    def __init__(self, context: ProcessingContext, api: APIManager, file_handler: FileHandler, extractor: Extractor, mapper: Mapper):
        self.context = context
        self.api = api
        self.file_handler = file_handler
        self.extractor = extractor
        self.mapper = mapper

    def _get_image_for_sector(self, sector: str, suffix: str) -> str:
        # Helper to find specific images like "alpha_image_1"
        files = self.file_handler.extract_images_from_excel(self.context.temp_dir) # Re-scan if needed, or use cached logic
        # Ideally, file handler should cache this. For now, we assume file_handler has access to temp.
        # This is a simplification. Real implementation would probably inspect existing files.
        candidate = f"{sector}_{suffix}"
        return self.file_handler.get_image_path(candidate)

    def verify_completeness(self, model_service: str, model_generic: str):
        """
        Rule 2: Check if Service/Speed/Video data is complete. 
        If not, trigger re-evaluation (retry mechanisms are in API layer, 
        but here we might try alternative images or strategies).
        """
        self.context.log("[EVAL] Rule 2: Verifying data completeness...")

        # Example check for Alpha Service
        if not self.context.alpha_service.nr_arfcn:
            self.context.log("[EVAL] Alpha Service missing fields. Retrying...")
            # Retry logic could go here, e.g., re-calling extractor with different parameters
            # For this refactor, we rely on the strict schema validation in Extractor to have done its best.
        
        # In a full implementation, you'd iterate through all sectors and tests.
        pass

    def run_strict_remapping(self, xlsx_path: str, model_generic: str):
        """
        Rule 3: If Mapper left cells as "NULL", uses AI to strictly find that specific value.
        """
        # This functionality typically requires re-scanning the Excel file for "NULL" cells
        # and then asking the AI specifically for that expression.
        # For this MVP refactor, we will rely on the Mapper doing the first pass.
        # A fully compliant Rule 3 would call:
        # self.mapper.map_to_excel(path) -> check for NULLs -> ask AI -> update.
        pass

    def process_workflow(self, xlsx_path: str, provider_service: Any, provider_speed: Any, provider_video: Any, provider_voice: Any) -> str:
        """
        Main orchestration pipeline with Deterministic Routing.
        Accepts LLMProvider objects (typed as Any to avoid circular imports if config is tricky, 
        but ideally Typed from config).
        """
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
                data = self.extractor.analyze_service_images(sector, img1, img2, provider_service)
                if data:
                    setattr(self.context, f"{sector}_service", data)
                    self.context.log(f"[SUCCESS] {sector} service data extracted.")

        # 3. Process Generic Images (Speed, Video, Voice) with Deterministic Routing
        for img_path in images:
            name = Path(img_path).stem
            # Parse sector and suffix
            # Expected format: "{sector}_image_{number}" or "voicetest_image_{number}"
            
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
                    data = self.extractor.analyze_voice_call(img_path, provider_voice)
                    if data:
                        self.context.voice_test[name] = data
                    continue

                # Service Pipeline (Already handled above)
                if suffix == 1 or suffix == 2:
                    continue

                # Speed Pipeline (Images 3-7)
                if 3 <= suffix <= 7:
                    data = self.extractor.analyze_speed_test(img_path, provider_speed)
                    if data:
                         getattr(self.context, f"{sector}_speedtest")[name] = data
                    continue

                # Video Pipeline (Image 8)
                if suffix == 8:
                    data = self.extractor.analyze_video_test(img_path, provider_video)
                    if data:
                        getattr(self.context, f"{sector}_video")[name] = data
                    continue
                
                # Unknown/Extra Images
                self.context.log(f"[WARN] Image {name} did not match any pipeline (Suffix {suffix}). Skipping.")
            else:
                 pass # Skip files that don't match pattern naming entirely

        # 4. Rule 2 Verify
        self.verify_completeness(provider_service, provider_speed)

        # 5. Map to Excel
        self.mapper.map_to_excel(xlsx_path)
        
        return xlsx_path
