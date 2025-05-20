import pytest
import json
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile

# Add project root to sys.path to allow importing backend and prompts
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend import app as fastapi_app # For potential future client testing
from backend import UPLOAD_DIR # To ensure it's created for tests if needed
# from prompts import ESG_EXTRACTION_PROMPT # ESG_EXTRACTION_PROMPT is used in backend.py, not directly in tests
from google.genai.types import Part, GenerateContentResponse

# Ensure UPLOAD_DIR exists for tests that might use it indirectly
UPLOAD_DIR.mkdir(exist_ok=True)

# Sample ESG data for mocking LLM responses
SAMPLE_SUCCESSFUL_ESG_DATA = {
  "environmental": {
    "climate_strategy_summary": "Committed to reducing emissions.",
    "ghg_emissions_scope1": 1000,
  },
  "social": {
    "employee_health_safety_ltir": 0.5,
  },
  "governance": {
    "board_independence_percentage": 75,
  },
  "overall_esg_commitments_summary": "Strong overall commitment."
}

MALFORMED_JSON_STRING = "{'environmental': {'climate_strategy_summary': 'Test'}, ...this is not valid json"

# --- Tests for backend.py ESG processing ---

@pytest.fixture
def mock_analysis_session_fixture(): # Renamed to avoid conflict with patch argument
    # Mocks the Gemini analysis session used in backend.py's /generate_report
    mock_session = MagicMock()
    mock_session.send_message = MagicMock()
    return mock_session

@patch('backend.client.chats.create') # Mocks the creation of the session
@patch('backend.generate_pdf_report') # Mock PDF generation to isolate ESG data processing
@patch('backend.LangChainHandler') # Mock LangChainHandler as it's not directly part of ESG LLM call
def test_generate_report_successful_esg_extraction(
    mock_lc_handler, mock_pdf_gen, mock_chats_create, tmp_path # mock_analysis_session_fixture is implicitly used via mock_chats_create
):
    # Test that /generate_report correctly processes a successful ESG extraction response.
    
    # Configure the mock session create to return our specific mock_analysis_session_fixture
    # Need to create an instance of the fixture here to configure it.
    mock_session_instance = MagicMock()
    mock_session_instance.send_message = MagicMock()
    mock_chats_create.return_value = mock_session_instance
    
    # Mock the financial extraction response (first call to send_message)
    mock_financial_response = MagicMock(spec=GenerateContentResponse)
    mock_financial_response.text = json.dumps({
        "company_name": "TestCorp", "reporting_period": "2023", "currency": "USD",
        "income_statement": {"net_sales": 1000}, "balance_sheet": {"total_assets": 500},
        "notes": {}
    })
    
    # Mock the ESG extraction response (should be the 4th call to send_message)
    mock_esg_response = MagicMock(spec=GenerateContentResponse)
    mock_esg_response.text = json.dumps(SAMPLE_SUCCESSFUL_ESG_DATA)
    
    # Mock other LLM calls (overview, findings)
    mock_overview_response = MagicMock(spec=GenerateContentResponse)
    mock_overview_response.text = "Business overview."
    mock_findings_response = MagicMock(spec=GenerateContentResponse)
    mock_findings_response.text = "Key findings."

    # Order of calls: financial, overview, findings, esg
    mock_session_instance.send_message.side_effect = [
        mock_financial_response, 
        mock_overview_response,  
        mock_findings_response,  
        mock_esg_response        
    ]
    
    dummy_file_content = b"dummy pdf content"
    dummy_file_path = tmp_path / "dummy.pdf"
    with open(dummy_file_path, "wb") as f:
        f.write(dummy_file_content)

    from fastapi.testclient import TestClient
    client = TestClient(fastapi_app)

    with open(dummy_file_path, "rb") as f:
        response = client.post("/generate_report", files={"file": ("dummy.pdf", f, "application/pdf")})

    assert response.status_code == 200 
    mock_pdf_gen.assert_called_once()
    args, kwargs = mock_pdf_gen.call_args
    report_data_arg = args[0]
    
    assert "extracted_esg_data" in report_data_arg
    assert report_data_arg["extracted_esg_data"] == SAMPLE_SUCCESSFUL_ESG_DATA

@patch('backend.client.chats.create')
@patch('backend.generate_pdf_report')
@patch('backend.LangChainHandler')
def test_generate_report_malformed_esg_json(
    mock_lc_handler, mock_pdf_gen, mock_chats_create, tmp_path
):
    # Tests that /generate_report handles malformed JSON from ESG extraction.
    mock_session_instance = MagicMock()
    mock_session_instance.send_message = MagicMock()
    mock_chats_create.return_value = mock_session_instance
    
    mock_financial_response = MagicMock(spec=GenerateContentResponse); 
    mock_financial_response.text = json.dumps({"income_statement": {"net_sales": 100}})
    mock_overview_response = MagicMock(spec=GenerateContentResponse); mock_overview_response.text = "Overview"
    mock_findings_response = MagicMock(spec=GenerateContentResponse); mock_findings_response.text = "Findings"
    
    mock_esg_response = MagicMock(spec=GenerateContentResponse)
    mock_esg_response.text = MALFORMED_JSON_STRING 
    
    mock_session_instance.send_message.side_effect = [
        mock_financial_response, mock_overview_response, mock_findings_response, mock_esg_response
    ]

    dummy_file_path = tmp_path / "dummy.pdf"
    dummy_file_path.write_bytes(b"test content")
    
    from fastapi.testclient import TestClient
    client = TestClient(fastapi_app)
    with open(dummy_file_path, "rb") as f:
        response = client.post("/generate_report", files={"file": ("dummy.pdf", f, "application/pdf")})

    assert response.status_code == 200 
    mock_pdf_gen.assert_called_once()
    args, kwargs = mock_pdf_gen.call_args
    report_data_arg = args[0]
    
    assert "extracted_esg_data" in report_data_arg
    assert "error" in report_data_arg["extracted_esg_data"]
    # Check for part of the specific error message from backend.py's parsing logic
    assert "Failed to parse ESG data" in report_data_arg["extracted_esg_data"]["error"]


@patch('backend.client.chats.create')
@patch('backend.generate_pdf_report')
@patch('backend.LangChainHandler')
def test_generate_report_esg_extraction_llm_error(
    mock_lc_handler, mock_pdf_gen, mock_chats_create, tmp_path
):
    # Tests that /generate_report handles an error during the ESG LLM call.
    mock_session_instance = MagicMock()
    mock_session_instance.send_message = MagicMock()
    mock_chats_create.return_value = mock_session_instance

    mock_financial_response = MagicMock(spec=GenerateContentResponse)
    mock_financial_response.text = json.dumps({"income_statement": {"net_sales": 100}})
    mock_overview_response = MagicMock(spec=GenerateContentResponse); mock_overview_response.text = "Overview"
    mock_findings_response = MagicMock(spec=GenerateContentResponse); mock_findings_response.text = "Findings"
    
    mock_session_instance.send_message.side_effect = [
        mock_financial_response, 
        mock_overview_response, 
        mock_findings_response, 
        RuntimeError("Simulated LLM API error for ESG") 
    ]

    dummy_file_path = tmp_path / "dummy.pdf"
    dummy_file_path.write_bytes(b"test content")

    from fastapi.testclient import TestClient
    client = TestClient(fastapi_app)
    
    with open(dummy_file_path, "rb") as f:
        response = client.post("/generate_report", files={"file": ("dummy.pdf", f, "application/pdf")})

    # Even with an internal error during ESG processing, the main endpoint might still return 200
    # because the error is caught and packaged into the ESG data part of the report.
    assert response.status_code == 200 
    
    mock_pdf_gen.assert_called_once()
    args, kwargs = mock_pdf_gen.call_args
    report_data_arg = args[0]
    
    assert "extracted_esg_data" in report_data_arg
    assert "error" in report_data_arg["extracted_esg_data"]
    assert "Simulated LLM API error for ESG" in report_data_arg["extracted_esg_data"]["error"]

```

# --- Tests for report_generator.py ESG section ---

from report_generator import generate_pdf_report # Already imported if backend is, but good for clarity

@pytest.fixture
def base_report_data():
    # Provides a minimal base structure for report_data.
    return {
        "business_overview": "Test overview.",
        "key_findings": "Test findings.",
        "extracted_data": { # Minimal financial data
            "company_name": "TestCo",
            "income_statement": {"net_sales": 100},
            "balance_sheet": {"total_assets": 100}
        },
        "calculated_ratios": {"Current Ratio": {"ratio_value": 1.5}},
        # extracted_esg_data will be added by each test
    }

def test_report_generator_with_full_esg_data(base_report_data, tmp_path):
    # Tests PDF generation with a complete set of ESG data.
    report_data = base_report_data.copy()
    report_data["extracted_esg_data"] = {
        "environmental": {
            "climate_strategy_summary": "Climate strategy exists.",
            "ghg_emissions_scope1": 1200,
            "renewable_energy_percentage": 15.5
        },
        "social": {
            "employee_health_safety_ltir": 0.8,
            "diversity_equity_inclusion_summary": "DEI programs in place."
        },
        "governance": {
            "board_independence_percentage": 80.0,
            "business_ethics_policy_summary": "Ethics policy updated."
        },
        "overall_esg_commitments_summary": "Company shows strong commitment to ESG principles."
    }
    
    output_pdf_path = tmp_path / "report_full_esg.pdf"
    
    try:
        generate_pdf_report(report_data, str(output_pdf_path))
    except Exception as e:
        pytest.fail(f"PDF generation failed with full ESG data: {e}")
    
    assert output_pdf_path.exists()
    assert output_pdf_path.stat().st_size > 0 # Check that file is not empty

def test_report_generator_with_partial_esg_data(base_report_data, tmp_path):
    # Tests PDF generation with some ESG data missing.
    report_data = base_report_data.copy()
    report_data["extracted_esg_data"] = {
        "environmental": {
            "climate_strategy_summary": "Partial climate strategy."
            # Other env fields missing
        },
        "social": {
            # Social category present but empty
        },
        "governance": None, # Entire governance category missing
        "overall_esg_commitments_summary": None # Overall summary missing
    }
    
    output_pdf_path = tmp_path / "report_partial_esg.pdf"
    
    try:
        generate_pdf_report(report_data, str(output_pdf_path))
    except Exception as e:
        pytest.fail(f"PDF generation failed with partial ESG data: {e}")
        
    assert output_pdf_path.exists()
    assert output_pdf_path.stat().st_size > 0

def test_report_generator_with_empty_esg_data(base_report_data, tmp_path):
    # Tests PDF generation when extracted_esg_data is empty.
    report_data = base_report_data.copy()
    report_data["extracted_esg_data"] = {} # Empty ESG data
    
    output_pdf_path = tmp_path / "report_empty_esg.pdf"
    
    try:
        generate_pdf_report(report_data, str(output_pdf_path))
    except Exception as e:
        pytest.fail(f"PDF generation failed with empty ESG data: {e}")

    assert output_pdf_path.exists()
    assert output_pdf_path.stat().st_size > 0

def test_report_generator_with_esg_error_message(base_report_data, tmp_path):
    # Tests PDF generation when ESG data contains an error message.
    report_data = base_report_data.copy()
    report_data["extracted_esg_data"] = {
        "error": "Failed to extract ESG data due to API timeout."
    }
    
    output_pdf_path = tmp_path / "report_esg_error.pdf"
    
    try:
        generate_pdf_report(report_data, str(output_pdf_path))
    except Exception as e:
        pytest.fail(f"PDF generation failed with ESG error message: {e}")

    assert output_pdf_path.exists()
    assert output_pdf_path.stat().st_size > 0

def test_report_generator_no_esg_data_key(base_report_data, tmp_path):
    # Tests PDF generation if 'extracted_esg_data' key is missing entirely.
    report_data = base_report_data.copy()
    # "extracted_esg_data" key is intentionally omitted
    
    output_pdf_path = tmp_path / "report_no_esg_key.pdf"
    
    try:
        generate_pdf_report(report_data, str(output_pdf_path))
    except Exception as e:
        pytest.fail(f"PDF generation failed when ESG key is missing: {e}")

    assert output_pdf_path.exists()
    assert output_pdf_path.stat().st_size > 0
