import os
from typing import List, Dict, Optional, Any, Set
from pydantic import BaseModel, Field
import streamlit as st

# ----------------- CONSTANTS -----------------
API_BASE = "https://integrate.api.nvidia.com/v1"
REASONING_MODEL_DEFAULT = "deepseek-ai/deepseek-v3.2"
VISION_MODEL_DEFAULT = "meta/llama-3.2-90b-vision-instruct"

# ----------------- PROVIDER CONFIG -----------------
class LLMProvider(BaseModel):
    name: str = "Default"
    base_url: str = "https://integrate.api.nvidia.com/v1"
    api_key: str
    model: str


# ----------------- SCHEMAS -----------------
# Strict Pydantic models to prevent hallucinations

class ServiceData(BaseModel):
    nr_arfcn: Optional[float] = None
    nr_band: Optional[float] = None
    nr_pci: Optional[float] = None
    nr_bw: Optional[float] = None
    nr5g_rsrp: Optional[float] = None
    nr5g_rsrq: Optional[float] = None
    nr5g_sinr: Optional[float] = None
    lte_band: Optional[float] = None
    lte_earfcn: Optional[float] = None
    lte_pci: Optional[float] = None
    lte_bw: Optional[float] = None
    lte_rsrp: Optional[float] = None
    lte_rsrq: Optional[float] = None
    lte_sinr: Optional[float] = None

class SpeedTestData(BaseModel):
    download_mbps: Optional[float] = None
    upload_mbps: Optional[float] = None
    ping_ms: Optional[float] = None
    jitter_ms: Optional[float] = None

class VideoTestData(BaseModel):
    max_resolution: Optional[str] = None
    load_time_ms: Optional[float] = None
    buffering_percentage: Optional[float] = None

class VoiceCallData(BaseModel):
    phone_number: Optional[str] = None
    call_duration_seconds: Optional[float] = None
    call_status: Optional[str] = None
    time: Optional[str] = None

# ----------------- STATE MANAGEMENT -----------------
class ProcessingContext:
    """
    Holds the state for a single processing run.
    Replaces global variables.
    """
    def __init__(self):
        # Service Data by Sector
        self.alpha_service: ServiceData = ServiceData()
        self.beta_service: ServiceData = ServiceData()
        self.gamma_service: ServiceData = ServiceData()

        # Generic Tests (Keyed by image name)
        self.alpha_speedtest: Dict[str, SpeedTestData] = {}
        self.beta_speedtest: Dict[str, SpeedTestData] = {}
        self.gamma_speedtest: Dict[str, SpeedTestData] = {}

        self.alpha_video: Dict[str, VideoTestData] = {}
        self.beta_video: Dict[str, VideoTestData] = {}
        self.gamma_video: Dict[str, VideoTestData] = {}

        self.voice_test: Dict[str, VoiceCallData] = {}

        # Derived Data
        self.average: Dict[str, Dict[str, Optional[float]]] = {}
        self.extract_text: List[str] = []

        # Operational State
        self.logs: List[str] = []
        self.retried_images: Set[str] = set()
        self.temp_dir: str = ""

    def log(self, message: str):
        """Append to internal log list."""
        import time
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self.logs.append(f"[{ts}] {message}")

    def to_dict(self):
        """Helper for expression resolution"""
        return {
            "alpha_service": self.alpha_service.model_dump(),
            "beta_service": self.beta_service.model_dump(),
            "gamma_service": self.gamma_service.model_dump(),
            "alpha_speedtest": {k: v.model_dump() for k, v in self.alpha_speedtest.items()},
            "beta_speedtest": {k: v.model_dump() for k, v in self.beta_speedtest.items()},
            "gamma_speedtest": {k: v.model_dump() for k, v in self.gamma_speedtest.items()},
            "alpha_video": {k: v.model_dump() for k, v in self.alpha_video.items()},
            "beta_video": {k: v.model_dump() for k, v in self.beta_video.items()},
            "gamma_video": {k: v.model_dump() for k, v in self.gamma_video.items()},
            "voice_test": {k: v.model_dump() for k, v in self.voice_test.items()},
            "average": self.average
        }

# Global Schema Definitions for AI Prompts
SERVICE_SCHEMA_JSON = ServiceData.model_json_schema()['properties']
SPEED_SCHEMA_JSON = SpeedTestData.model_json_schema()['properties']
VIDEO_SCHEMA_JSON = VideoTestData.model_json_schema()['properties']
VOICE_SCHEMA_JSON = VoiceCallData.model_json_schema()['properties']


