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
    st.set_page_config(page_title="Cellular Processor", layout="wide")
    st.title("🔬 Advanced Cellular Template Processor")
    st.write("**Dual-Model Pipeline** | Vision + Reasoning | No Nulls")

    # Session State Init
    if "context" not in st.session_state:
        st.session_state.context = ProcessingContext()
    if "last_log_count" not in st.session_state:
        st.session_state.last_log_count = 0
    
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
        st.session_state.last_log_count = 0
        st.rerun()

    # Main Area
    if st.session_state.get("api_valid"):
        uploaded_file = st.file_uploader("📁 Upload .xlsx Template", type=["xlsx"])
        
        st.subheader("⚙️ Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**👁️ Vision Model** (OCR)")
            vision_model = st.text_input("Vision Model", value=VISION_MODEL_DEFAULT, key="vision_m")
            
        with col2:
            st.markdown("**🧠 Reasoning Model** (JSON Parsing)")
            reasoning_model = st.text_input("Reasoning Model", value=REASONING_MODEL_DEFAULT, key="reason_m")

        # Create providers
        provider_vision = None
        provider_reasoning = None
        
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

        if uploaded_file and provider_vision and provider_reasoning:
            if st.button("🚀 Start Processing", type="primary"):
                # Setup
                temp_dir = tempfile.mkdtemp()
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Create fresh context
                ctx = ProcessingContext()
                st.session_state.context = ctx
                st.session_state.last_log_count = 0

                api_mgr = APIManager(st.session_state.token)
                file_handler = FileHandler(ctx)
                file_handler.setup_temp_dir(temp_dir)
                
                extractor = Extractor(api_mgr, ctx)
                mapper = Mapper(ctx, api_mgr)
                evaluator = Evaluator(ctx, api_mgr, file_handler, extractor, mapper)

                # Live log display
                log_container = st.empty()
                status_container = st.empty()
                
                status_container.info("🔄 Processing... Please wait")
                
                try:
                    # Run workflow
                    out_path = evaluator.process_workflow(
                        file_path, 
                        provider_vision,
                        provider_reasoning
                    )
                    
                    status_container.success("✅ Processing Complete!")
                    
                    if out_path:
                        with open(out_path, "rb") as f:
                            st.download_button(
                                "📥 Download Result", 
                                f, 
                                file_name="processed_output.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                except Exception as e:
                    status_container.error(f"❌ Error: {e}")
                    ctx.log(f"CRITICAL ERROR: {e}")
                    import traceback
                    ctx.log(traceback.format_exc())

    # Logs Section - Always visible
    st.markdown("---")
    st.subheader("📋 Processing Logs")
    
    log_filter = st.selectbox("Filter", ["All", "Errors Only", "Success Only"], index=0)
    
    logs = st.session_state.context.logs
    if log_filter == "Errors Only":
        logs = [l for l in logs if "ERROR" in l or "WARN" in l]
    elif log_filter == "Success Only":
        logs = [l for l in logs if "OK" in l or "SUCCESS" in l]
    
    # Show logs
    log_text = "\n".join(logs) if logs else "No logs yet. Upload a file and start processing."
    st.text_area("Logs", value=log_text, height=400, key="log_display")
    
    # Auto-refresh hint
    if logs:
        st.caption(f"📊 Total: {len(st.session_state.context.logs)} log entries")

if __name__ == "__main__":
    main()
