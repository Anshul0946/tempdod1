import os
import io
import shutil
import openpyxl
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional
from .config import ProcessingContext
from .api_manager import APIManager

class FileHandler:
    def __init__(self, context: ProcessingContext):
        self.context = context

    def setup_temp_dir(self, temp_dir: str):
        self.context.temp_dir = temp_dir
        os.makedirs(os.path.join(temp_dir, "images"), exist_ok=True)

    def extract_images_from_excel(self, xlsx_path: str) -> List[str]:
        """
        Extracts images from Excel and saves them to temp/images.
        Returns list of saved image paths.
        """
        self.context.log(f"Analyzing template file: {xlsx_path}")
        try:
            wb = openpyxl.load_workbook(xlsx_path)
            sheet = wb.active # Assuming active sheet
        except Exception as e:
            self.context.log(f"[ERROR] Could not open Excel file: {e}")
            return []

        images = getattr(sheet, "_images", [])
        if not images:
            self.context.log("[WARN] No images found in workbook.")
            return []

        # Sort images by location (top-left to bottom-right)
        images_with_locations = []
        for image in images:
            try:
                row = image.anchor._from.row + 1
                col = image.anchor._from.col
            except Exception:
                row, col = 0, 0
            images_with_locations.append({"image": image, "row": row, "col": col})

        images_sorted = sorted(images_with_locations, key=lambda i: (i["row"], i["col"]))
        
        saved_paths = []
        output_folder = os.path.join(self.context.temp_dir, "images")
        
        # Heuristic naming based on column index (legacy logic)
        counters = {"alpha": 0, "beta": 0, "gamma": 0, "voicetest": 0, "unknown": 0}
        
        for itm in images_sorted:
            col_index = itm["col"]
            if 0 <= col_index < 4:
                sector = "alpha"
            elif 4 <= col_index < 8:
                sector = "beta"
            elif 8 <= col_index < 12:
                sector = "gamma"
            elif 12 <= col_index < 18:
                sector = "voicetest"
            else:
                sector = "unknown"
                
            counters[sector] += 1
            filename = f"{sector}_image_{counters[sector]}.png"
            out_path = os.path.join(output_folder, filename)
            
            try:
                img_data = itm["image"]._data()
                pil = Image.open(io.BytesIO(img_data))
                pil.save(out_path, "PNG")
                saved_paths.append(out_path)
                self.context.log(f"  - Extracted {filename}")
            except Exception as e:
                self.context.log(f"[ERROR] Failed to save {filename}: {e}")

        return saved_paths

    def get_image_path(self, name: str) -> Optional[str]:
        """Simple lookup for an image in the temp folder."""
        path = os.path.join(self.context.temp_dir, "images", f"{name}.png")
        if os.path.exists(path):
            return path
        return None
