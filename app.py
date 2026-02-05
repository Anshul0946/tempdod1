import streamlit as st
import tempfile
import os
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

def main():
    st.set_page_config(page_title="Cellular Processor (Dual-Model)", layout="wide")
    st.title("🔬 Advanced Cellular Template Processor")
    st.write("**Dual-Model Architecture** | Vision + Reasoning | Fault Tolerant")

    # Session State Init
    if "context" not in st.session_state:
        st.session_state.context = ProcessingContext()
    
    # Sidebar
    st.sidebar.header("🔑 API Configuration")
    token_input = st.sidebar.text_input("API Key", type="password")
    
    if st.sidebar.button("Validate Key"):
        api = APIManager(token_input)
        if api.validate_api_key():
            st.session_state.api_valid = True
            st.session_state.token = token_input
            st.sidebar.success("✅ Valid Key")
        else:
            st.sidebar.error("❌ Invalid Key")

    if st.sidebar.button("🔄 Reset State"):
        st.session_state.context = ProcessingContext()
        st.rerun()

    # Main Area
    if st.session_state.get("api_valid"):
        uploaded_file = st.file_uploader("📁 Upload .xlsx Template", type=["xlsx"])
        
        st.subheader("⚙️ Model Configuration")
        
        # Two main models
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**👁️ Vision Model** (Image Text Extraction)")
            vision_model = st.text_input("Vision Model", value=VISION_MODEL_DEFAULT, key="vision_m")
            st.caption("Reads raw text from images")
            
        with col2:
            st.markdown("**🧠 Reasoning Model** (JSON Parsing & Validation)")
            reasoning_model = st.text_input("Reasoning Model", value=REASONING_MODEL_DEFAULT, key="reason_m")
            st.caption("Structures text into JSON, handles merging")

        # Create providers
        if st.session_state.get("token"):
            provider_vision = LLMProvider(
                name="Vision",
                api_key=st.session_state.token,
                model=vision_model,
                base_url=API_BASE
            )
            
            provider_reasoning = LLMProvider(
                name="Reasoning", 
                api_key=st.session_state.token,
                model=reasoning_model,
                base_url=API_BASE
            )

        if uploaded_file:
            if st.button("🚀 Start Processing", type="primary"):
                # Setup
                temp_dir = tempfile.mkdtemp()
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Initialization
                ctx = ProcessingContext()
                ctx.log("=" * 50)
                ctx.log("INITIALIZING DUAL-MODEL PIPELINE")
                ctx.log("=" * 50)
                st.session_state.context = ctx

                api_mgr = APIManager(st.session_state.token)
                file_handler = FileHandler(ctx)
                file_handler.setup_temp_dir(temp_dir)
                
                extractor = Extractor(api_mgr, ctx)
                mapper = Mapper(ctx, api_mgr)
                evaluator = Evaluator(ctx, api_mgr, file_handler, extractor, mapper)

                # Execution
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with st.spinner("Processing with Dual-Model Pipeline..."):
                    try:
                        status_text.text("Stage 1: Extracting images from Excel...")
                        progress_bar.progress(10)
                        
                        out_path = evaluator.process_workflow(
                            file_path, 
                            provider_vision,
                            provider_reasoning
                        )
                        
                        progress_bar.progress(100)
                        status_text.text("Complete!")
                        
                        if out_path:
                            st.success("✅ Processing Complete!")
                            with open(out_path, "rb") as f:
                                st.download_button(
                                    "📥 Download Result", 
                                    f, 
                                    file_name="processed_output.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                    except Exception as e:
                        st.error(f"❌ Critical Error: {e}")
                        ctx.log(f"CRITICAL ERROR: {e}")
                        import traceback
                        ctx.log(traceback.format_exc())

    # Logs Section
    st.markdown("---")
    st.subheader("📋 Live Logs")
    
    # Add filter
    log_filter = st.selectbox("Filter", ["All", "Errors Only", "Success Only"], index=0)
    
    logs = st.session_state.context.logs
    if log_filter == "Errors Only":
        logs = [l for l in logs if "ERROR" in l or "WARN" in l]
    elif log_filter == "Success Only":
        logs = [l for l in logs if "SUCCESS" in l]
    
    st.text_area("Logs", value="\n".join(logs), height=400)

if __name__ == "__main__":
    main()
