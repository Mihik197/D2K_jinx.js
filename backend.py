from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
from typing import List, Optional, Dict, Any
import uuid
from pathlib import Path
import logging
import tempfile
import json
import re
from fastapi.responses import FileResponse
from report_generator import generate_pdf_report
from prompts import EXTRACTION_PROMPT
import time

load_dotenv()

# Configure the Gemini API using the newer library's client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production!
    allow_credentials=True,
    allow_methods=["*"],  # Adjust for production!
    allow_headers=["*"],
)

UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Enhanced session storage that tracks documents and message history
chat_sessions = {}  # session_id -> chat object
session_documents = {}  # session_id -> list of document parts
session_history = {}  # session_id -> list of messages

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


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        # Initialize chat session if needed
        if session_id not in chat_sessions:
            chat_sessions[session_id] = client.chats.create(model='gemini-2.0-flash')
            session_history[session_id] = []
            session_documents[session_id] = []
        
        chat_session = chat_sessions[session_id]
        user_message_content = request.messages[-1].content if request.messages else ""
        
        # Prepare message parts: first any documents in context, then the new message
        message_parts = []
        
        # Add any documents associated with this session
        if session_documents.get(session_id):
            message_parts.extend(session_documents[session_id])
        
        # Add the user's text message
        message_parts.append(types.Part.from_text(text=user_message_content))
        
        # Send the message with all context
        if len(message_parts) == 1:
            # Just a text message, no documents
            response = chat_session.send_message(user_message_content)
        else:
            # Message with document context
            response = chat_session.send_message(message_parts)
        
        # Store in history
        session_history[session_id].append({"role": "user", "content": user_message_content})
        session_history[session_id].append({"role": "assistant", "content": response.text})
        
        return ChatResponse(response=response.text, session_id=session_id)
    
    except Exception as e:
        logging.exception("Error in /chat endpoint:")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_document", response_model=ChatResponse)
async def upload_document(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    session_id: Optional[str] = Form(None)
):
    file_path = None
    try:
        session_id = session_id or str(uuid.uuid4())
        
        # Save uploaded file
        file_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        logging.info(f"File saved to {file_path} ({len(content)} bytes)")
        
        # Determine MIME type
        mime_type = "application/pdf"  # default
        if file.filename.lower().endswith(".csv"):
            mime_type = "text/csv"
        elif file.filename.lower().endswith((".xlsx", ".xls")):
            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        logging.info(f"Using MIME type: {mime_type} for {file.filename}")
        
        # Read file data
        with open(file_path, "rb") as f:
            file_data = f.read()
        
        # Create document part
        document_part = types.Part.from_bytes(
            data=file_data,
            mime_type=mime_type
        )
        
        # Initialize chat session if needed
        if session_id not in chat_sessions:
            chat_sessions[session_id] = client.chats.create(model='gemini-2.0-flash')
            session_history[session_id] = []
            session_documents[session_id] = []
        
        # Store document for future context
        session_documents[session_id].append(document_part)
        
        # Create the message with document and prompt
        message_parts = [document_part, types.Part.from_text(text=prompt)]
        
        # Send the message
        chat_session = chat_sessions[session_id]
        response = chat_session.send_message(message_parts)
        
        # Store in history
        session_history[session_id].append({
            "role": "user", 
            "content": f"[Uploaded document: {file.filename}] {prompt}"
        })
        session_history[session_id].append({"role": "assistant", "content": response.text})
        
        # Clean up file
        if file_path:
            file_path.unlink(missing_ok=True)
        
        return ChatResponse(response=response.text, session_id=session_id)
    
    except Exception as e:
        if file_path:
            file_path.unlink(missing_ok=True)
        logging.exception("Error in upload_document endpoint:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    if session_id not in session_history:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"session_id": session_id, "history": session_history[session_id]}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/generate_report")
async def generate_report(file: UploadFile = File(...)):
    """
    Endpoint to analyze a financial document and generate a PDF report.
    Directly uses the uploaded PDF file with Gemini's vision capabilities.
    # """
    file_path = None
    try:
        # Save uploaded file temporarily
        file_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        logging.info(f"File saved to {file_path} ({len(content)} bytes)")
        
        # Determine MIME type
        mime_type = "application/pdf"  # default
        if file.filename.lower().endswith(".csv"):
            mime_type = "text/csv"
        elif file.filename.lower().endswith((".xlsx", ".xls")):
            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        
        # Create a temporary chat session for analysis
        analysis_session = client.chats.create(model='gemini-2.0-flash')
        
        # Read file data
        with open(file_path, "rb") as f:
            file_data = f.read()
        
        # Create document part
        document_part = types.Part.from_bytes(
            data=file_data,
            mime_type=mime_type
        )
        
        # Use EXTRACTION_PROMPT from prompts.py instead of inline text
        extraction_prompt = EXTRACTION_PROMPT
        # Send the extraction request with document context
        message_parts = [document_part, types.Part.from_text(text=extraction_prompt)]
        
        # Add retry logic for the model
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                extraction_response = analysis_session.send_message(message_parts)
                break  # If successful, break the retry loop
            except Exception as e:
                if "503" in str(e) and "overloaded" in str(e) and attempt < max_retries-1:
                    logging.warning(f"Gemini API overloaded, retrying in {retry_delay} seconds (attempt {attempt+1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    # If we've exhausted retries or it's a different error, re-raise
                    raise
        
        # Process the JSON response
        json_text = extraction_response.text
        try:
            # First try direct JSON parsing
            extracted_data = json.loads(json_text)
        except json.JSONDecodeError:
            # If that fails, try to extract JSON from code blocks
            json_match = re.search(r'```json\s*(.*?)\s*```', json_text, re.DOTALL)
            if json_match:
                extracted_data = json.loads(json_match.group(1))
            else:
                # One more attempt without code block markers
                json_match = re.search(r'({[\s\S]*})', json_text)
                if json_match:
                    extracted_data = json.loads(json_match.group(1))
                else:
                    raise ValueError("Failed to extract valid JSON from LLM response")
        
        # Calculate financial ratios
        from langchain_integration import LangChainHandler
        handler = LangChainHandler()
        calculated_ratios = handler.calculate_financial_ratios(extracted_data)
        
        # Generate business overview using Gemini
        overview_prompt = f"""Based on the following extracted financial data, provide a concise business overview. It should be a couple paragraphs long and cover the company's financial health, performance, and any notable trends or insights. Try to focus also on the management's discussion and analysis of the financial results:
                {json.dumps(extracted_data, indent=2)}

                Output only the business overview text."""
        
        # Add retry logic for overview generation
        for attempt in range(max_retries):
            try:
                overview_response = analysis_session.send_message(overview_prompt)
                business_overview = overview_response.text.strip()
                break
            except Exception as e:
                if "503" in str(e) and "overloaded" in str(e) and attempt < max_retries-1:
                    logging.warning(f"Gemini API overloaded during overview generation, retrying in {retry_delay} seconds (attempt {attempt+1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    # If it's not a 503/overloaded error or we've exhausted retries, provide a fallback
                    logging.error(f"Error generating business overview: {str(e)}")
                    business_overview = "Business overview could not be generated due to AI service limitations. Please review the extracted financial data for insights."
                    break
        
        # Generate key findings using Gemini
        findings_prompt = f"""Analyze the following extracted financial data and calculated ratios, and provide key findings 
                with focus on profitability, liquidity, solvency, and any notable trends. Each should be at least one paragraph and have high quality content:

                Extracted Data:
                {json.dumps(extracted_data, indent=2)}

                Calculated Ratios:
                {json.dumps(calculated_ratios, indent=2)}

                Output only the key findings text."""
        
        # Add retry logic for findings generation
        retry_delay = 2  # Reset delay
        for attempt in range(max_retries):
            try:
                findings_response = analysis_session.send_message(findings_prompt)
                key_findings = findings_response.text.strip()
                break
            except Exception as e:
                if "503" in str(e) and "overloaded" in str(e) and attempt < max_retries-1:
                    logging.warning(f"Gemini API overloaded during findings generation, retrying in {retry_delay} seconds (attempt {attempt+1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    # If it's not a 503/overloaded error or we've exhausted retries, provide a fallback
                    logging.error(f"Error generating key findings: {str(e)}")
                    key_findings = "Key findings could not be generated due to AI service limitations. Please review the extracted financial data and calculated ratios for insights."
                    break
        
        # Create report data
        report_data = {
            "business_overview": business_overview,
            "key_findings": key_findings,
            "extracted_data": extracted_data,
            "calculated_ratios": calculated_ratios
        }
        
        # Generate PDF report in a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            output_path = tmp.name
        
        generate_pdf_report(report_data, output_path)
        
        # Clean up the temporary file
        if file_path:
            file_path.unlink(missing_ok=True)
        
        return FileResponse(path=output_path, filename="financial_report.pdf", media_type="application/pdf")
    except Exception as e:
        # Clean up on error
        if file_path:
            file_path.unlink(missing_ok=True)
        logging.exception("Error generating report")
        
        # Provide more helpful error message based on the exception
        if "503" in str(e) and "overloaded" in str(e):
            error_message = "The AI service is currently overloaded. Please try again in a few minutes."
        else:
            error_message = str(e)
            
        raise HTTPException(status_code=500, detail=error_message)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)