import streamlit as st
import httpx
import json
from typing import List, Dict
import time
import os
import pkg_resources

# Check for Streamlit version to handle rerun function appropriately
st_version = pkg_resources.get_distribution("streamlit").version
use_rerun = True  # Default to using st.rerun() for newer versions

backend_url = os.environ.get("BACKEND_URL", "https://my-financial-app-backend.onrender.com")

st.set_page_config(
    page_title="Financial Statement Analyzer",
    page_icon="📊",
    layout="wide"
)

# Set up the page structure
st.title("📊 Financial Statement Analysis Chat")
st.markdown("Upload your financial documents (PDFs, CSVs, Excel) and ask questions to get instant analysis.")

# Add instructions for the user
with st.expander("How to use this app"):
    st.markdown("""
    **Getting Started:**
    1. Upload a financial document using the sidebar (supports PDF, CSV, Excel)
    2. Ask a question about the document in the chat
    3. For the initial question, provide context such as: "Analyze this financial statement" or "What are the key financial metrics?"

    **Example Questions:**
    - "Analyze this financial statement and provide key metrics"
    - "What is the company's current ratio?"
    - "Calculate profitability ratios based on this data"
    - "What are the trends in revenue growth?"
    - "Identify any red flags in this financial statement"
    """)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "documents" not in st.session_state:
    st.session_state.documents = []

# -------------------------------
# Document upload section in sidebar
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader(
        "Upload a financial document (PDF, CSV, Excel)",
        type=["pdf", "csv", "xlsx", "xls"],
        help="Upload a financial document to analyze"
    )

    if uploaded_file is not None:
        st.success(f"File '{uploaded_file.name}' is ready for analysis")

    # Show all documents in the current session
    if st.session_state.documents:
        st.header("Uploaded Documents")
        for doc in st.session_state.documents:
            st.info(f"📄 {doc}")

    st.header("About")
    st.markdown("This is a prototype for an AI-driven Financial Statement Analysis Platform.")
    st.markdown("You can chat with the AI or upload documents for analysis.")

    # Add a button to clear the conversation
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.session_state.documents = []
        st.rerun()

    # Add the team credit message here
    st.markdown("---") # Optional separator
    st.markdown("Built by **Team Jinx.js**:")
    st.markdown("- Mihik Chaudhari")
    st.markdown("- Bhavya Goyal")
    st.markdown("- Krishna Naudiyal")


# -------------------------------
# New section for Detailed Analysis & Report Generation at the top
st.header("Detailed Analysis & Report Generation")
# If a file has been uploaded, we do not force pasting text.
if not uploaded_file:
    document_text = st.text_area("Paste the full document text for analysis (if not using file upload):", height=200)
else:
    st.info("Using uploaded document for analysis.")
    document_text = ""  # We rely on uploaded file and its stored context.

if st.button("Run Analysis"):
    # Require an uploaded file for analysis since we want to leverage Gemini’s native PDF vision.
    if not uploaded_file:
        st.error("Please upload a document for analysis.")
    else:
        with st.spinner("Generating analysis and report..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = httpx.post(
                    f"{backend_url}/generate_report",
                    files=files,
                    timeout=120.0
                )
                if response.status_code == 200:
                    pdf_bytes = response.content
                    st.success("Analysis complete!")
                    st.download_button(
                        label="Download Report PDF",
                        data=pdf_bytes,
                        file_name="financial_report.pdf",
                        mime="application/pdf"
                    )
                    # Show a placeholder for inline analysis details.
                    analysis_details = (
                        "Business Overview:\n[Extracted overview text...]\n\n"
                        "Key Findings:\n[Extracted key findings...]\n\n"
                        "Extracted Data:\n[JSON with extracted financial data...]\n\n"
                        "Calculated Ratios:\n[JSON with calculated ratios...]"
                    )
                    with st.expander("View Analysis Details (Placeholder - Download PDF for full report)"):
                        st.text(analysis_details)
                else:
                    st.error(f"Error generating report: {response.status_code}\n{response.text}")
            except httpx.ConnectError as ce:
                st.error(f"Connection Error: Could not connect to the backend service at {backend_url}. Please ensure the backend is running.")
            except Exception as e:
                st.error(f"Exception during analysis: {str(e)}")

# -------------------------------
# Chat interface (moved to the bottom)
st.header("Chat with the AI")

# Function to handle regular chat
def process_chat(prompt):
    try:
        with st.spinner("Thinking..."):
            response = httpx.post(
                f"{backend_url}/chat",
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "session_id": st.session_state.session_id
                },
                timeout=60.0
            )
            if response.status_code == 200:
                data = response.json()
                st.session_state.session_id = data["session_id"]
                return data["response"]
            else:
                st.error(f"Error: {response.text}")
                return f"Error communicating with backend: {response.status_code}"
    except httpx.ConnectError as ce:
         st.error(f"Connection Error: Could not connect to the backend service at {backend_url}. Please ensure the backend is running.")
         return f"Connection Error: Could not connect to the backend at {backend_url}"
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return f"Error: {str(e)}"

# Function to handle document upload and chat
def process_document_chat(file, prompt):
    try:
        with st.spinner("Processing document and analyzing..."):
            files = {"file": (file.name, file.getvalue(), "application/octet-stream")}
            form_data = {
                "prompt": prompt,
                "session_id": st.session_state.session_id if st.session_state.session_id else "",
            }
            response = httpx.post(
                f"{backend_url}/upload_document",
                files=files,
                data=form_data,
                timeout=120.0
            )
            if response.status_code == 200:
                data = response.json()
                st.session_state.session_id = data["session_id"]
                if file.name not in st.session_state.documents:
                    st.session_state.documents.append(file.name)
                return data["response"]
            else:
                st.error(f"Error: {response.text}")
                return f"Error processing document: {response.status_code}"
    except httpx.ConnectError as ce:
         st.error(f"Connection Error: Could not connect to the backend service at {backend_url}. Please ensure the backend is running.")
         return f"Connection Error: Could not connect to the backend at {backend_url}"
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return f"Error: {str(e)}"

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input section
prompt = st.chat_input("Ask a question about financial statements...")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # If a new file was uploaded earlier and not already processed, use it
    if uploaded_file and uploaded_file.name not in st.session_state.documents:
        response_text = process_document_chat(uploaded_file, prompt)
        # Re-run to update the sidebar with the new document name immediately
        st.rerun()
    else:
        response_text = process_chat(prompt)

    with st.chat_message("assistant"):
        st.markdown(response_text)
    st.session_state.messages.append({"role": "assistant", "content": response_text})

    # Re-run only if it wasn't a document upload chat, to avoid double rerun
    if not (uploaded_file and uploaded_file.name in st.session_state.documents):
         st.rerun()