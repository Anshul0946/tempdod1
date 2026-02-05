import streamlit as st
import tempfile
import os
import shutil
from pathlib import Path
from core.config import ProcessingContext, API_BASE, MODEL_SERVICE_DEFAULT, MODEL_GENERIC_DEFAULT
from core.api_manager import APIManager
from core.file_handler import FileHandler
from core.extractor import Extractor
from core.mapper import Mapper
from core.evaluator import Evaluator

def main():
    st.set_page_config(page_title="Cellular Processor (Refactored)", layout="wide")
    st.title("Advanced Cellular Template Processor")
    st.write("Refactored Modular Architecture | Fault Tolerant | Strict Validation")

    # Session State Init
    if "context" not in st.session_state:
        st.session_state.context = ProcessingContext()
    
    # Sidebar
    st.sidebar.header("Configuration")
    token_input = st.sidebar.text_input("API Key", type="password")
    
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
        st.experimental_rerun()

    # Main Area
    if st.session_state.get("api_valid"):
        uploaded_file = st.file_uploader("Upload .xlsx Template", type=["xlsx"])
        
        col1, col2 = st.columns(2)
        model_service = col1.selectbox("Service Model", [MODEL_SERVICE_DEFAULT])
        model_generic = col2.selectbox("Generic Model", [MODEL_GENERIC_DEFAULT])

        if uploaded_file:
            if st.button("Start Processing", type="primary"):
                # Setup
                temp_dir = tempfile.mkdtemp()
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Initialization
                ctx = ProcessingContext()
                ctx.log("Initializing modules...")
                st.session_state.context = ctx # Update view with fresh context

                api_mgr = APIManager(st.session_state.token)
                file_handler = FileHandler(ctx)
                file_handler.setup_temp_dir(temp_dir)
                
                extractor = Extractor(api_mgr, ctx)
                mapper = Mapper(ctx, api_mgr)
                evaluator = Evaluator(ctx, api_mgr, file_handler, extractor, mapper)

                # Execution
                with st.spinner("Processing..."):
                   try:
                       out_path = evaluator.process_workflow(file_path, model_service, model_generic)
                       if out_path:
                           st.success("Processing Complete!")
                           with open(out_path, "rb") as f:
                               st.download_button("Download Result", f, file_name="processed_output.xlsx")
                   except Exception as e:
                       st.error(f"Critical Error: {e}")
                       ctx.log(f"CRITICAL ERROR: {e}")

    # Logs
    st.markdown("---")
    st.subheader("Live Logs")
    st.text_area("Logs", value="\n".join(st.session_state.context.logs), height=400)

if __name__ == "__main__":
    main()
