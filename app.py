# app.py
import streamlit as st
import httpx
import json
from typing import List, Dict
import time
import os

st.set_page_config(
    page_title="Financial Statement Analyzer",
    page_icon="📊",
    layout="wide"
)

# --- Configuration ---
# Use environment variables for backend URL in production
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
REPORT_ENDPOINT = f"{BACKEND_URL}/generate_report"
CHAT_ENDPOINT = f"{BACKEND_URL}/chat"
UPLOAD_ENDPOINT = f"{BACKEND_URL}/upload_document"

# --- Page Setup ---
st.title("📊 Financial Statement Analysis Chat")
st.markdown("""
Upload your financial documents (PDF, CSV, Excel) for analysis.
- **Generate a Full Report:** Use the 'Generate Full PDF Report' button below.
- **Chat:** Ask specific questions about the uploaded document(s) using the chat interface.
""")

# Instructions Expander
with st.expander("How to use this app"):
    st.markdown("""
    **Getting Started:**
    1.  **Upload:** Use the sidebar to upload a financial document (PDF, CSV, XLSX).
    2.  **Generate Report:** Click "Generate Full PDF Report" for a comprehensive analysis PDF.
    3.  **Chat:** Ask questions in the chat box. The AI will use the context of uploaded documents in the current session.

    **Example Questions (Chat):**
    - "What is the company's current ratio based on the uploaded document?"
    - "Summarize the key profitability trends."
    - "Are there any liquidity concerns mentioned?"
    - "Explain the main points from the business overview section of the report."
    """)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = [] # Stores chat history: [{"role": "user/assistant", "content": "..."}]
if "session_id" not in st.session_state:
    st.session_state.session_id = None # Stores unique ID for backend session tracking
if "uploaded_file_info" not in st.session_state:
    # Stores info about the latest successfully processed file for context
    # e.g., {"name": "report.pdf", "processed_for_report": True/False, "processed_for_chat": True/False}
    st.session_state.uploaded_file_info = None
if "active_docs" not in st.session_state:
    # Tracks names of documents successfully processed by the /upload_document endpoint in this session
    st.session_state.active_docs = []


# --- Sidebar ---
with st.sidebar:
    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader(
        "Upload PDF, CSV, or Excel file",
        type=["pdf", "csv", "xlsx", "xls"],
        key="file_uploader" # Assign a key for stability
    )

    if uploaded_file is not None:
        # Display message immediately upon selection
        st.info(f"Selected: '{uploaded_file.name}'. Ready for report generation or chat.")
        # Store minimal info temporarily, actual processing happens on button click or chat input
        st.session_state.uploaded_file_info = {"name": uploaded_file.name}


    # Display documents active in the current chat session
    st.header("Active Session Documents")
    if st.session_state.active_docs:
        for doc_name in st.session_state.active_docs:
            st.success(f"✓ {doc_name}")
    else:
        st.caption("No documents processed in chat yet.")


    st.header("⚙️ Session Management")
    if st.button("Clear Conversation & Session"):
        # Reset all relevant session state variables
        st.session_state.messages = []
        st.session_state.session_id = None
        st.session_state.uploaded_file_info = None
        st.session_state.active_docs = []
        st.success("Conversation and session cleared.")
        # Use experimental_rerun for a cleaner reset
        st.experimental_rerun()

    st.header("ℹ️ About")
    st.markdown("AI-driven financial statement analysis prototype. Chat or generate reports.")


# --- Report Generation Section ---
st.header("📈 Generate Full PDF Report")

# Only show report generation button if a file has been selected in the uploader
if uploaded_file:
    if st.button(f"Generate Full PDF Report for '{uploaded_file.name}'", key="generate_report_button"):
        if uploaded_file is not None:
            st.session_state.uploaded_file_info = {"name": uploaded_file.name, "processed_for_report": False} # Mark for processing
            with st.spinner(f"Generating comprehensive analysis report for '{uploaded_file.name}'... This may take a minute."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = httpx.post(
                        REPORT_ENDPOINT,
                        files=files,
                        timeout=180.0 # Increased timeout for potentially long analysis
                    )

                    if response.status_code == 200:
                        pdf_bytes = response.content
                        st.success("✅ Analysis Report Generated!")
                        st.download_button(
                            label=f"Download Report: {uploaded_file.name.replace('.', '_')}_Report.pdf",
                            data=pdf_bytes,
                            file_name=f"{uploaded_file.name.replace('.', '_')}_Report.pdf",
                            mime="application/pdf",
                            key="download_button"
                        )
                        # Mark file as processed for report context
                        st.session_state.uploaded_file_info["processed_for_report"] = True

                        # Placeholder for analysis details - backend currently only returns PDF
                        # We could add another endpoint to fetch the analysis JSON data separately if needed.
                        analysis_details_placeholder = (
                            "Report generated successfully. Download the PDF above for full details.\n\n"
                            "(To display analysis details here, the backend would need to return the structured data "
                            "in addition to the PDF, or provide a separate endpoint to fetch it.)"
                        )
                        with st.expander("Analysis Details (Placeholder)"):
                            st.info(analysis_details_placeholder)

                    else:
                        try:
                            error_detail = response.json().get("detail", response.text)
                        except json.JSONDecodeError:
                            error_detail = response.text
                        st.error(f"❌ Error generating report (Status {response.status_code}):\n{error_detail}")
                        st.session_state.uploaded_file_info["processed_for_report"] = False # Mark as failed

                except httpx.RequestError as req_err:
                     st.error(f"❌ Network Error: Could not connect to backend at {BACKEND_URL}. Is it running?\n{req_err}")
                     st.session_state.uploaded_file_info["processed_for_report"] = False
                except Exception as e:
                    st.error(f"❌ An unexpected error occurred: {str(e)}")
                    st.session_state.uploaded_file_info["processed_for_report"] = False
        else:
            st.warning("Please upload a document first using the sidebar.")
else:
    st.info("Upload a document using the sidebar to enable report generation.")


# --- Chat Interface Section ---
st.header("💬 Chat with AI Analyzer")

# Display chat messages from history
if "messages" in st.session_state:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input field
prompt = st.chat_input("Ask a question about the uploaded document(s)...")

if prompt:
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Determine if a new file needs to be uploaded/processed for chat context
    # Check if the file selected in the uploader is different from the last one processed or not yet processed
    needs_upload_processing = False
    current_file_name = uploaded_file.name if uploaded_file else None
    if current_file_name and current_file_name not in st.session_state.active_docs:
         needs_upload_processing = True

    # Call backend
    try:
        with st.spinner("Thinking..."):
            assistant_response = "Sorry, something went wrong." # Default error message
            target_endpoint = UPLOAD_ENDPOINT if needs_upload_processing else CHAT_ENDPOINT

            # Prepare payload
            if needs_upload_processing and uploaded_file:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                data = {
                     "prompt": prompt,
                     "session_id": st.session_state.session_id or "" # Send current session ID if exists
                }
                response = httpx.post(target_endpoint, files=files, data=data, timeout=120.0)
            else:
                payload = {
                     "messages": [{"role": "user", "content": prompt}], # Send only the latest user message
                     "session_id": st.session_state.session_id
                }
                response = httpx.post(target_endpoint, json=payload, timeout=60.0)

            # Process response
            if response.status_code == 200:
                data = response.json()
                assistant_response = data["response"]
                st.session_state.session_id = data["session_id"] # Update session ID
                # If upload was successful, add doc name to active list
                if needs_upload_processing and current_file_name:
                    st.session_state.active_docs.append(current_file_name)
                    st.info(f"'{current_file_name}' added to chat context for this session.") # Inform user
                    # Rerun to update the sidebar display
                    st.experimental_rerun()
            else:
                try:
                    error_detail = response.json().get("detail", response.text)
                except json.JSONDecodeError:
                    error_detail = response.text
                assistant_response = f"Error: Failed to get response from backend (Status {response.status_code}).\n{error_detail}"
                st.error(assistant_response)

    except httpx.RequestError as req_err:
         assistant_response = f"Error: Could not connect to the analysis service.\n{req_err}"
         st.error(assistant_response)
    except Exception as e:
        assistant_response = f"Error: An unexpected error occurred.\n{str(e)}"
        st.error(assistant_response)

    # Add assistant response to chat history and display it
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    with st.chat_message("assistant"):
        st.markdown(assistant_response)

# Trigger rerun if active docs changed (e.g., after successful upload via chat)
# This ensures the sidebar updates promptly. (Already handled by rerun in the processing block)