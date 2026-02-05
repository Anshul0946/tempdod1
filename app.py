import streamlit as st
import tempfile
import os
import shutil
from pathlib import Path
from core.config import (
    ProcessingContext, API_BASE, LLMProvider
)
from core.api_manager import APIManager
from core.file_handler import FileHandler
from core.extractor import Extractor
from core.mapper import Mapper
from core.evaluator import Evaluator

def main():
    st.set_page_config(page_title="Cellular Processor (Refactored)", layout="wide")
    st.title("Advanced Cellular Template Processor")
    st.write("Refactored Modular Architecture | Fault Tolerant | Strict Separation | Multi-Provider")

    # Session State Init
    if "context" not in st.session_state:
        st.session_state.context = ProcessingContext()
    
    # Sidebar
    st.sidebar.header("Configuration")
    token_input = st.sidebar.text_input("API Key (Default)", type="password")
    
    if st.sidebar.button("Validate Key"):
        api = APIManager(token_input)
        if api.validate_api_key():
            st.session_state.api_valid = True
            st.session_state.token = token_input
            st.sidebar.success("Valid Key")
        else:
            st.sidebar.error("Invalid Key")

    if st.sidebar.button("Reset State"):
        st.session_state.context = ProcessingContext()
        st.rerun()

    # Main Area
    if st.session_state.get("api_valid"):
        uploaded_file = st.file_uploader("Upload .xlsx Template", type=["xlsx"])
        
        st.subheader("Pipeline Configuration")
        st.info("Currently using Default API Key. In future, each pipeline can have distinct API URLs/Keys.")
        
        c1, c2 = st.columns(2)
        c3, c4 = st.columns(2)
        
        # Default Models for now
        DEFAULT_MODEL = "meta/llama-3.2-90b-vision-instruct"
        
        with c1:
            st.markdown("**Service Pipeline** (Images 1-2)")
            srv_model = st.text_input("Service Model", value=DEFAULT_MODEL, key="srv_m")
            # In future: srv_url = ... srv_key = ...
            provider_service = LLMProvider(
                name="Service", 
                api_key=st.session_state.token, 
                model=srv_model,
                base_url=API_BASE
            )
            
        with c2:
            st.markdown("**Speed Pipeline** (Images 3-7)")
            spd_model = st.text_input("Speed Model", value=DEFAULT_MODEL, key="spd_m")
            provider_speed = LLMProvider(
                name="Speed", 
                api_key=st.session_state.token, 
                model=spd_model,
                base_url=API_BASE
            )

        with c3:
            st.markdown("**Video Pipeline** (Image 8)")
            vid_model = st.text_input("Video Model", value=DEFAULT_MODEL, key="vid_m")
            provider_video = LLMProvider(
                name="Video", 
                api_key=st.session_state.token, 
                model=vid_model,
                base_url=API_BASE
            )

        with c4:
            st.markdown("**Voice Pipeline** (Voicetest)")
            voi_model = st.text_input("Voice Model", value=DEFAULT_MODEL, key="voi_m")
            provider_voice = LLMProvider(
                name="Voice", 
                api_key=st.session_state.token, 
                model=voi_model,
                base_url=API_BASE
            )

        if uploaded_file:
            if st.button("Start Processing", type="primary"):
                # Setup
                temp_dir = tempfile.mkdtemp()
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Initialization
                ctx = ProcessingContext()
                ctx.log("Initializing pipelines...")
                st.session_state.context = ctx # Update view with fresh context

                api_mgr = APIManager(st.session_state.token)
                file_handler = FileHandler(ctx)
                file_handler.setup_temp_dir(temp_dir)
                
                extractor = Extractor(api_mgr, ctx)
                mapper = Mapper(ctx, api_mgr)
                evaluator = Evaluator(ctx, api_mgr, file_handler, extractor, mapper)

                # Execution
                with st.spinner("Processing with Strict Separation..."):
                   try:
                       out_path = evaluator.process_workflow(
                           file_path, 
                           provider_service, 
                           provider_speed, 
                           provider_video, 
                           provider_voice
                        )
                       if out_path:
                           st.success("Processing Complete!")
                           with open(out_path, "rb") as f:
                               st.download_button("Download Result", f, file_name="processed_output.xlsx")
                   except Exception as e:
                       st.error(f"Critical Error: {e}")
                       ctx.log(f"CRITICAL ERROR: {e}")
                       import traceback
                       ctx.log(traceback.format_exc())

    # Logs
    st.markdown("---")
    st.subheader("Live Logs")
    st.text_area("Logs", value="\n".join(st.session_state.context.logs), height=400)

if __name__ == "__main__":
    main()
