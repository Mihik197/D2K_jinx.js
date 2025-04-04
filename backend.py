# backend.py

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse # Added JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import uuid
from pathlib import Path
import logging
import tempfile
import json
import re
import pandas as pd
import asyncio # Needed for running async utils
import numpy as np # For numerical operations

# Local imports
from report_generator import generate_pdf_report, NumberedCanvas # Import canvas for TOC
from analysis_utils import ( # Import new analysis functions
    extract_financial_data, calculate_financial_ratios,
    detect_financial_red_flags, generate_business_overview,
    generate_key_findings, generate_sentiment_analysis,
    generate_business_model_ideas
)
# Prompts are now used within analysis_utils
# from prompts import EXTRACTION_PROMPT

load_dotenv()

# Configure the Gemini API client (ensure GOOGLE_API_KEY is set)
try:
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
except Exception as e:
    logging.error(f"Failed to initialize Gemini client: {e}")
    # Handle appropriately - maybe exit or raise a specific startup error
    raise RuntimeError(f"Failed to initialize Gemini client: {e}") from e


app = FastAPI(title="Financial Statement Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# TODO: Replace in-memory storage with persistent solution (Redis, DB) for production
chat_sessions = {}  # session_id -> chat object (google.generativeai.generative_models.ChatSession)
session_documents = {}  # session_id -> list of document parts (types.Part)
session_history = {}  # session_id -> list of messages ({role: str, content: str})

logging.basicConfig(level=logging.INFO)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str

# --- Chat Endpoints ---

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    user_message_content = request.messages[-1].content if request.messages else ""

    try:
        # Initialize chat session if needed
        if session_id not in chat_sessions:
            # Use the model specified for chat, e.g., gemini-2.0-flash
            # TODO: Consider adding system instructions here if needed for the chat persona
            chat_sessions[session_id] = client.chats.create(model='gemini-2.0-flash')
            session_history[session_id] = []
            session_documents[session_id] = [] # Initialize document list for session
            logging.info(f"Created new chat session: {session_id}")

        chat_session = chat_sessions[session_id]

        # Prepare message parts:
        # TODO: Refine context strategy. Appending *all* docs can exceed limits.
        # Consider only using the last N documents or a specific active document.
        message_parts = []
        if session_documents.get(session_id):
            message_parts.extend(session_documents[session_id])
            logging.info(f"Session {session_id}: Including {len(session_documents[session_id])} documents in context.")

        message_parts.append(types.Part.from_text(text=user_message_content))

        # Send message using the ChatSession object
        logging.info(f"Session {session_id}: Sending message with {len(message_parts)} parts.")
        response = await chat_session.send_message_async(message_parts) # Use async

        # Store history
        session_history[session_id].append({"role": "user", "content": user_message_content})
        session_history[session_id].append({"role": "assistant", "content": response.text})

        return ChatResponse(response=response.text, session_id=session_id)

    except Exception as e:
        logging.exception(f"Error in /chat endpoint for session {session_id}:")
        # Consider providing a more user-friendly error message
        raise HTTPException(status_code=500, detail=f"An internal error occurred during chat processing: {str(e)}")


@app.post("/upload_document", response_model=ChatResponse)
async def upload_document(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    session_id: Optional[str] = Form(None)
):
    session_id = session_id or str(uuid.uuid4())
    file_content = await file.read()
    if not file_content:
         raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Determine MIME type (more robustly if possible)
    mime_type_map = {
        ".pdf": "application/pdf",
        ".csv": "text/csv",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xls": "application/vnd.ms-excel",
        # Add more types if needed
    }
    file_ext = Path(file.filename).suffix.lower()
    mime_type = mime_type_map.get(file_ext, "application/octet-stream") # Default binary
    logging.info(f"Detected MIME type: {mime_type} for file: {file.filename}")

    try:
         # Create document part directly from bytes
        document_part = types.Part.from_bytes(
            data=file_content,
            mime_type=mime_type
        )

        # Initialize session if it doesn't exist
        if session_id not in chat_sessions:
            chat_sessions[session_id] = client.chats.create(model='gemini-2.0-flash')
            session_history[session_id] = []
            session_documents[session_id] = []
            logging.info(f"Created new chat session {session_id} during file upload.")

        # Store document part for session context
        # TODO: Decide on strategy - replace old docs or append? Appending for now.
        session_documents.setdefault(session_id, []).append(document_part)
        logging.info(f"Added document '{file.filename}' to session {session_id}. Total docs: {len(session_documents[session_id])}")

        # Create the message with document and prompt
        message_parts = [document_part, types.Part.from_text(text=prompt)]

        # Send the message via the chat session
        chat_session = chat_sessions[session_id]
        logging.info(f"Session {session_id}: Sending message with uploaded document '{file.filename}' and prompt.")
        response = await chat_session.send_message_async(message_parts) # Use async

        # Store history (indicating document upload)
        history_prompt = f"[Uploaded Document: {file.filename}]\n{prompt}"
        session_history[session_id].append({"role": "user", "content": history_prompt})
        session_history[session_id].append({"role": "assistant", "content": response.text})

        return ChatResponse(response=response.text, session_id=session_id)

    except Exception as e:
        logging.exception(f"Error in /upload_document for session {session_id}:")
        raise HTTPException(status_code=500, detail=f"An internal error occurred processing the document: {str(e)}")


@app.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    # Basic check, consider more robust session validation
    if session_id not in session_history:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "history": session_history.get(session_id, [])}


@app.get("/health")
async def health_check():
    # TODO: Add deeper health checks (e.g., can connect to Gemini API?)
    return {"status": "healthy"}


# --- Analysis and Report Generation Endpoint ---

@app.post("/generate_report")
async def generate_report_endpoint(file: UploadFile = File(...)):
    """
    Analyzes an uploaded financial document (PDF, CSV, XLSX) and generates
    a detailed PDF report along with the analysis data in JSON format.
    """
    logging.info(f"Received request to generate report for file: {file.filename}")
    file_content = await file.read()
    if not file_content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Determine MIME type
    mime_type_map = {".pdf": "application/pdf", ".csv": "text/csv", ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", ".xls": "application/vnd.ms-excel"}
    file_ext = Path(file.filename).suffix.lower()
    mime_type = mime_type_map.get(file_ext, "application/octet-stream")
    logging.info(f"Processing '{file.filename}' with MIME type: {mime_type}")

    output_pdf_path = None
    report_data = {} # Initialize report data dict

    try:
        # 1. Create Document Part
        document_part = types.Part.from_bytes(data=file_content, mime_type=mime_type)

        # 2. Extract Financial Data
        extracted_data = await extract_financial_data(client, document_part)
        if not extracted_data: # Check if extraction failed (returned None)
            logging.error("Critical failure: Financial data extraction failed.")
            raise HTTPException(status_code=500, detail="Failed to extract essential financial data from the document. Cannot proceed.")
        report_data["extracted_data"] = extracted_data # Store even if potentially incomplete

        # 3. Calculate Ratios
        calculated_ratios = calculate_financial_ratios(extracted_data)
        report_data["calculated_ratios"] = calculated_ratios # Store ratios

        # 4. Detect Red Flags (Python Rules)
        python_red_flags_result = detect_financial_red_flags(extracted_data, calculated_ratios)
        python_flags = python_red_flags_result.get("red_flags", [])

        # 5. Generate LLM-based Insights (Concurrently)
        logging.info("Starting concurrent LLM analysis tasks...")
        overview_task = generate_business_overview(client, extracted_data)
        findings_task = generate_key_findings(client, extracted_data, calculated_ratios)
        sentiment_task = generate_sentiment_analysis(client, extracted_data)
        model_ideas_task = generate_business_model_ideas(client, extracted_data)

        # Await all tasks
        overview_result, findings_result, sentiment_result, model_ideas_result = await asyncio.gather(
            overview_task, findings_task, sentiment_task, model_ideas_task
        )
        logging.info("Finished concurrent LLM analysis tasks.")

        # --- Process results carefully, handling potential errors/Nones ---

        # Business Overview (String expected)
        report_data["business_overview"] = overview_result if isinstance(overview_result, str) else "Analysis unavailable."
        if "Error:" in report_data["business_overview"]: logging.warning("Business Overview generation failed.")

        # Key Findings (Dict expected with specific keys)
        key_findings_analysis = findings_result.get("key_findings_analysis", {})
        llm_flags = findings_result.get("identified_red_flags", [])
        # Combine different parts of findings into one string for the report for now
        # TODO: Could refine PDF to show findings subsections
        findings_summary = key_findings_analysis.get("executive_summary", "Summary not available.")
        profit_analysis = key_findings_analysis.get("profitability", "Profitability analysis not available.")
        liq_solv_analysis = key_findings_analysis.get("liquidity_solvency", "Liquidity/Solvency analysis not available.")
        effic_analysis = key_findings_analysis.get("efficiency", "Efficiency analysis not available.")
        trends_analysis = key_findings_analysis.get("trends_anomalies", "Trends/Anomalies analysis not available.")
        # Combine into a single string or keep structured? For PDF simplicity now, combine.
        report_data["key_findings"] = (
            f"**Executive Summary:**\n{findings_summary}\n\n"
            f"**Profitability:**\n{profit_analysis}\n\n"
            f"**Liquidity & Solvency:**\n{liq_solv_analysis}\n\n"
            f"**Efficiency:**\n{effic_analysis}\n\n"
            f"**Trends & Anomalies:**\n{trends_analysis}"
        )
        if "Analysis failed" in findings_summary or "Error:" in findings_summary: logging.warning("Key Findings generation failed.")

        # Sentiment Analysis (Dict expected)
        sentiment_analysis_content = sentiment_result.get("sentiment_analysis", {})
        # Format into a string for the report
        sa_tone = sentiment_analysis_content.get('overall_tone', 'N/A')
        sa_indicators = sentiment_analysis_content.get('key_indicators', 'N/A')
        sa_outlook = sentiment_analysis_content.get('forward_outlook_assessment', 'N/A')
        sa_risk = sentiment_analysis_content.get('risk_disclosure_level', 'N/A')
        report_data["sentiment_analysis"] = (
            f"**Overall Tone:** {sa_tone}\n"
            f"**Key Indicators:** {sa_indicators}\n"
            f"**Forward Outlook:** {sa_outlook}\n"
            f"**Risk Disclosure Level:** {sa_risk}"
        )
        if "Error" in sa_tone or "failed" in sa_indicators: logging.warning("Sentiment Analysis generation failed.")

        # Business Model Ideas (Dict expected)
        model_suggestions = model_ideas_result.get("business_model_suggestions", [])
        feasibility_notes = model_ideas_result.get("feasibility_notes", "N/A")
        ideas_str_list = []
        if model_suggestions:
             for i, idea in enumerate(model_suggestions):
                  ideas_str_list.append(
                      f"**Suggestion {i+1}: {idea.get('suggestion_name', 'N/A')}**\n"
                      f"- Concept: {idea.get('concept', 'N/A')}\n"
                      f"- Rationale: {idea.get('rationale', 'N/A')}\n"
                      f"- Potential Impact: {idea.get('potential_impact', 'N/A')}\n"
                      f"- Key Risks: {idea.get('key_risks', 'N/A')}"
                  )
             report_data["business_model"] = "\n\n".join(ideas_str_list) + f"\n\n**Feasibility Notes:** {feasibility_notes}"
        else:
             report_data["business_model"] = f"No specific suggestions generated. Feasibility Notes: {feasibility_notes}"
        if "failed" in feasibility_notes or "Error:" in feasibility_notes : logging.warning("Business Model generation failed.")


        # Combine Red Flags
        all_red_flags = python_flags + llm_flags
        report_data["red_flags"] = all_red_flags

        log = logging.getLogger(__name__)
        # << --- ADD THIS LOGGING --- >>
        log.info("--- Final Report Data ---")
        # Log key parts safely
        log.info(f"Company Name: {report_data.get('extracted_data', {}).get('company_name')}")
        log.info(f"Overview Length: {len(report_data.get('business_overview', ''))}")
        log.info(f"Key Findings Snippet: {report_data.get('key_findings', '')[:200]}...")
        log.info(f"Sentiment Snippet: {report_data.get('sentiment_analysis', '')[:200]}...")
        log.info(f"Business Model Snippet: {report_data.get('business_model', '')[:200]}...")
        log.info(f"Calculated Ratios Keys: {list(report_data.get('calculated_ratios', {}).keys())}")
        log.info(f"Number of Red Flags: {len(report_data.get('red_flags', []))}")
        # Optionally log the full dict if not too large, or specific problematic fields
        # import pprint
        # log.debug(f"Full report_data:\n{pprint.pformat(report_data)}")
        log.info("--- End Final Report Data ---")
        # << --- END LOGGING --- >>


        # 6. Generate PDF Report
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            output_pdf_path = tmp.name
        log.info(f"Generating PDF report at: {output_pdf_path}")
        # Make sure the correct report_data dict is passed
        generate_pdf_report(report_data, output_pdf_path, canvas_maker=NumberedCanvas)
        log.info("PDF report generated successfully.")

        # 7. Return PDF File Response
        report_filename = f"{extracted_data.get('company_name', 'Financial').replace(' ', '_')}_Report.pdf"
        return FileResponse(
            path=output_pdf_path,
            filename=report_filename,
            media_type="application/pdf",
            background=os.remove(output_pdf_path) # Clean up temp file
       )

    except HTTPException as http_exc:
         # Re-raise HTTP exceptions directly
         raise http_exc
    except Exception as e:
        # Clean up temp file on any other error
        if output_pdf_path and os.path.exists(output_pdf_path):
            try:
                os.remove(output_pdf_path)
            except OSError:
                logging.warning(f"Could not remove temp PDF file: {output_pdf_path}")
        logging.exception("Error during report generation:")
        raise HTTPException(status_code=500, detail=f"An internal error occurred generating the report: {str(e)}")


# --- Anomaly Detection Endpoint (Kept as is) ---
@app.post("/detect_anomalies")
async def detect_anomalies(
    file: UploadFile = File(...),
    sensitivity: float = Form(1.0) # Default sensitivity
):
    """
    Detect anomalies in financial data (CSV expected) using statistical methods.
    """
    temp_file_path = None
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file provided for anomaly detection.")

        # Requires CSV format for Pandas
        if not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Anomaly detection currently only supports CSV files.")

        # Save content to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
             tmp.write(content)
             temp_file_path = tmp.name

        logging.info(f"Processing CSV for anomalies: {temp_file_path}")
        # Read financial data using Pandas
        try:
            financial_data = pd.read_csv(temp_file_path)
        except Exception as e:
             logging.error(f"Failed to read CSV file: {e}")
             raise HTTPException(status_code=400, detail=f"Could not parse CSV file: {e}")


        # Basic statistical anomaly detection (Z-score like)
        anomalies = []
        numeric_cols = financial_data.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
             return {"num_anomalies": 0, "anomalies": [], "message": "No numeric columns found to analyze."}

        logging.info(f"Analyzing numeric columns: {numeric_cols}")
        for column in numeric_cols:
            # Skip columns with no variance
            if financial_data[column].std() == 0 or financial_data[column].isnull().all():
                continue

            mean = financial_data[column].mean()
            std = financial_data[column].std()
            # Define threshold based on sensitivity (e.g., 1.0 -> 2 std devs, 1.5 -> 3 std devs)
            threshold = sensitivity * 2 * std # Adjust multiplier as needed

            if threshold == 0: continue # Avoid issues if std dev is zero somehow

            # Find rows where values deviate significantly
            outliers = financial_data[abs(financial_data[column] - mean) > threshold]

            for idx, row in outliers.iterrows():
                 deviation = abs(row[column] - mean)
                 # Simple confidence score based on deviation relative to 3 std devs
                 confidence = min(deviation / (3 * std), 1.0) if (3*std) > 0 else 1.0
                 anomalies.append({
                    "RowIndex": idx, # Use index instead of assuming 'Year'
                    "Column": column,
                    "Value": row[column],
                    "Mean": mean,
                    "Threshold": threshold,
                    "Deviation": deviation,
                    "Confidence": round(confidence, 3)
                 })

        logging.info(f"Found {len(anomalies)} anomalies with sensitivity {sensitivity}.")
        return {
            "num_anomalies": len(anomalies),
            "anomalies": anomalies
        }

    except HTTPException as http_exc:
        raise http_exc # Re-raise client errors
    except Exception as e:
        logging.exception("Error during anomaly detection processing:")
        raise HTTPException(status_code=500, detail=f"An internal error occurred during anomaly detection: {str(e)}")
    finally:
        # Clean up temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logging.info(f"Cleaned up temp file: {temp_file_path}")
            except OSError:
                 logging.warning(f"Could not remove temp anomaly file: {temp_file_path}")


# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    # Consider adding reload=True for development, but not for production
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)