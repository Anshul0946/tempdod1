import streamlit as st
import tempfile
import os
import threading
import time
from pathlib import Path
from core.config import (
    ProcessingContext, API_BASE, LLMProvider,
    VISION_MODEL_DEFAULT, REASONING_MODEL_DEFAULT
)
from core.api_manager import APIManager
from core.file_handler import FileHandler
from core.extractor import Extractor
from core.mapper import Mapper
from core.evaluator import Evaluator


class StreamlitLogger:
    """Custom logger that writes to both context and a live display."""
    def __init__(self, context: ProcessingContext, log_placeholder):
        self.context = context
        self.placeholder = log_placeholder
        
    def log(self, message: str):
        self.context.log(message)
        # Update display immediately
        self.placeholder.text_area(
            "Live Logs", 
            value="\n".join(self.context.logs[-50:]),  # Show last 50 logs
            height=300,
            key=f"log_{len(self.context.logs)}"
        )


def main():
    st.set_page_config(page_title="Cellular Processor", layout="wide")
    st.title("🔬 Cellular Template Processor")
    st.caption("Vision + Reasoning Pipeline")

    # Session State
    if "context" not in st.session_state:
        st.session_state.context = ProcessingContext()
    
    # Sidebar
    st.sidebar.header("🔑 API Key")
    token = st.sidebar.text_input("Enter API Key", type="password")
    
    if st.sidebar.button("Validate"):
        if token and len(token) > 20:
            st.session_state.api_valid = True
            st.session_state.token = token
            st.sidebar.success("✅ Valid")
        else:
            st.sidebar.error("❌ Invalid")

    if st.sidebar.button("🔄 Reset"):
        st.session_state.context = ProcessingContext()
        st.rerun()

    # Main Area
    if st.session_state.get("api_valid"):
        uploaded = st.file_uploader("📁 Upload Excel Template", type=["xlsx"])
        
        col1, col2 = st.columns(2)
        with col1:
            vision = st.text_input("Vision Model", value=VISION_MODEL_DEFAULT)
        with col2:
            reasoning = st.text_input("Reasoning Model", value=REASONING_MODEL_DEFAULT)

        if uploaded:
            if st.button("🚀 Process", type="primary"):
                # Setup
                temp_dir = tempfile.mkdtemp()
                file_path = os.path.join(temp_dir, uploaded.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded.getbuffer())

                # Create context
                ctx = ProcessingContext()
                st.session_state.context = ctx

                # Providers
                pv = LLMProvider(name="Vision", api_key=st.session_state.token, 
                                 model=vision, base_url=API_BASE)
                pr = LLMProvider(name="Reasoning", api_key=st.session_state.token,
                                 model=reasoning, base_url=API_BASE)

                # Components
                api = APIManager(st.session_state.token)
                fh = FileHandler(ctx)
                fh.setup_temp_dir(temp_dir)
                ext = Extractor(api, ctx)
                mapper = Mapper(ctx, api)
                ev = Evaluator(ctx, api, fh, ext, mapper)

                # Progress display
                progress = st.progress(0, text="Starting...")
                log_area = st.empty()
                
                try:
                    ctx.log("Starting processing...")
                    progress.progress(5, text="Extracting images...")
                    
                    # Run workflow
                    result = ev.process_workflow(file_path, pv, pr)
                    
                    progress.progress(100, text="Complete!")
                    
                    if result:
                        st.success("✅ Done!")
                        with open(result, "rb") as f:
                            st.download_button("📥 Download", f, 
                                             file_name="output.xlsx",
                                             mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    ctx.log(f"ERROR: {e}")

    # Always show logs
    st.markdown("---")
    st.subheader("📋 Logs")
    
    logs = st.session_state.context.logs
    if logs:
        st.text_area("Processing Logs", value="\n".join(logs), height=400)
        st.caption(f"{len(logs)} entries")
    else:
        st.info("Upload a file and click Process to see logs")

if __name__ == "__main__":
    main()
