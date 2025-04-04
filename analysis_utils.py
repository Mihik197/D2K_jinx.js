import json
import re
import logging
from typing import Dict, Any, List, Optional, Union # Added Union
from google import genai
from google.genai import types
import asyncio # Import asyncio here

# Import calculation functions
from financial_tools import (
    calculate_current_ratio, calculate_debt_to_equity_ratio,
    calculate_gross_margin_ratio, calculate_operating_margin_ratio,
    calculate_return_on_assets_ratio, calculate_return_on_equity_ratio,
    calculate_asset_turnover_ratio, calculate_inventory_turnover_ratio,
    calculate_receivables_turnover_ratio, calculate_debt_ratio,
    calculate_interest_coverage_ratio, safe_calculate
)
# Import prompts
from prompts import (
    EXTRACTION_PROMPT, OVERVIEW_PROMPT, FINDINGS_PROMPT,
    SENTIMENT_PROMPT, BUSINESS_MODEL_PROMPT
)

logging.basicConfig(level=logging.INFO)

# --- Configuration ---
ANALYSIS_MODEL_NAME = "gemini-1.5-flash-latest" # Or use 1.5-pro if needed/available
EXTRACTION_MODEL_NAME = "gemini-1.5-flash-latest" # Use a capable model for extraction
DEFAULT_TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = 8192

# --- Utility Functions ---

def parse_llm_json_response(response_text: str) -> Optional[Union[Dict, List]]: # Return type hint
    """
    Attempts to parse JSON from the LLM response, handling potential markdown code blocks.
    Returns None if parsing fails.
    """
    if not response_text:
        logging.warning("Cannot parse empty response text.")
        return None

    # 1. Look for ```json ... ```
    match_json = re.search(r'```json\s*([\s\S]*?)\s*```', response_text, re.DOTALL)
    if match_json:
        json_str = match_json.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON from ```json block: {e}\nContent: {json_str[:500]}...")
            # Fall through to try other methods if specific block fails

    # 2. Look for ``` ... ``` (generic code block)
    match_generic = re.search(r'```\s*([\s\S]*?)\s*```', response_text, re.DOTALL)
    if match_generic:
        potential_json_str = match_generic.group(1).strip()
        # Check if it looks like JSON before trying to parse
        if potential_json_str.startswith(("{", "[")) and potential_json_str.endswith(("}", "]")):
            try:
                return json.loads(potential_json_str)
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse JSON from generic ``` block: {e}\nContent: {potential_json_str[:500]}...")
                # Fall through

    # 3. Try parsing the whole response_text directly (if no code blocks found or parsing failed)
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        logging.warning("Direct JSON parsing of the whole response failed.")
        # Fall through

    # 4. Final attempt: Look for the first '{' or '[' and try to parse from there
    first_bracket = -1
    first_curly = -1
    try:
        first_bracket = response_text.index('[')
    except ValueError:
        pass
    try:
        first_curly = response_text.index('{')
    except ValueError:
        pass

    start_index = -1
    if first_bracket != -1 and first_curly != -1:
        start_index = min(first_bracket, first_curly)
    elif first_bracket != -1:
        start_index = first_bracket
    elif first_curly != -1:
        start_index = first_curly

    if start_index != -1:
        potential_json_substring = response_text[start_index:]
        # Attempt to find matching end bracket/curly - this is complex and error-prone
        # For simplicity, we'll just try parsing the substring
        try:
            # A more robust approach would involve balancing brackets/braces
            return json.loads(potential_json_substring)
        except json.JSONDecodeError:
            logging.error("Failed to parse potential JSON substring starting from first bracket/brace.")
            pass # Fall through

    logging.error("No valid JSON could be parsed from the LLM response.")
    return None


async def generate_gemini_content(
    client: genai.Client,
    model_name: str,
    prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = MAX_OUTPUT_TOKENS,
    document_part: Optional[types.Part] = None
) -> Optional[str]:
    """Calls the Gemini API. Returns generated text or None on error."""
    try:
        contents = []
        if document_part:
            contents.append(document_part)
        contents.append(types.Part.from_text(text=prompt))

        generation_config = types.GenerationConfig(
            # Ensure response is JSON object for prompts that require it
            response_mime_type="application/json",
            max_output_tokens=max_tokens,
            temperature=temperature
        )

        logging.info(f"Sending request to Gemini model: {model_name}...")
        response = await client.models.generate_content_async(
            model=model_name,
            contents=contents,
            generation_config=generation_config
        )
        logging.info(f"Received response from {model_name}.")

        # Handle potential blocks and empty parts
        if response.parts:
             # Accessing response.text should work even with application/json mime type
             # as the SDK handles the parsing internally for the .text attribute
             return response.text
        elif response.prompt_feedback and response.prompt_feedback.block_reason:
             block_reason_str = str(response.prompt_feedback.block_reason)
             logging.error(f"Content generation blocked. Reason: {block_reason_str}")
             # Return an error structure that can be parsed as JSON
             return json.dumps({"error": f"Content generation blocked due to safety reasons ({block_reason_str})."})
        else:
             logging.warning("Received response with no parts from Gemini API.")
             return json.dumps({"error": "Received empty response from model."})

    except Exception as e:
        logging.exception(f"Error calling Gemini API ({model_name}): {e}")
        # Return an error structure that can be parsed as JSON
        return json.dumps({"error": f"API call failed: {str(e)}"})


# --- Core Analysis Functions ---

async def extract_financial_data(
    client: genai.Client,
    document_part: types.Part
) -> Optional[Dict]:
    """Extracts structured financial data from a document using LLM."""
    logging.info("Step 1: Extracting financial data...")
    prompt = EXTRACTION_PROMPT
    response_text = await generate_gemini_content(
        client=client,
        model_name=EXTRACTION_MODEL_NAME,
        prompt=prompt,
        temperature=0.1, # Lower temp for extraction
        document_part=document_part
    )

    if not response_text:
        logging.error("Extraction failed: No response text from LLM.")
        return None # Indicate failure clearly

    logging.info(f"Raw extraction response text: {response_text[:300]}...")
    extracted_data = parse_llm_json_response(response_text)

    if isinstance(extracted_data, dict):
        logging.info("Successfully extracted and parsed financial data.")
        # Basic validation: check for essential keys
        if "company_name" not in extracted_data or "income_statement" not in extracted_data:
             logging.warning("Extracted JSON seems incomplete. Missing essential keys.")
             # You might return None or the partial data depending on requirements
        return extracted_data
    else:
        logging.error("Extraction failed: Could not parse valid JSON from response.")
        return None # Indicate failure


# --- Ratio Calculation (Remains the same, uses extracted_data dict) ---
def calculate_financial_ratios(data: Dict) -> Dict:
    # ... (keep the existing calculate_financial_ratios function as is) ...
    # Ensure it handles None values gracefully if extraction fails partially
    logging.info("Step 2: Calculating financial ratios...")
    if not data or not isinstance(data, dict):
         logging.warning("Cannot calculate ratios: Invalid input data.")
         return {"error": "Invalid input data for ratio calculation"}

    income_statement = data.get('income_statement', {}) or {}
    balance_sheet = data.get('balance_sheet', {}) or {}
    cash_flow = data.get('cash_flow', {}) or {}

    # Safely get values, defaulting to 0 if None or missing
    def get_val(source, key):
        v = source.get(key)
        # Check explicitly for None before trying float conversion
        if v is None:
             return 0.0
        try:
            # Handle potential string numbers with commas
            if isinstance(v, str):
                 v = v.replace(',', '')
            return float(v)
        except (ValueError, TypeError):
            logging.warning(f"Could not convert value for key '{key}' to float: {v}")
            return 0.0

    # ... rest of the get_val calls and ratio calculations ...
    # (Code is the same as previous version, ensure robustness to 0.0 defaults)

    # Example: Recalculating safe_calculate call for Current Ratio
    current_assets = get_val(balance_sheet, 'current_assets')
    current_liabilities = get_val(balance_sheet, 'current_liabilities')
    ratios = {
         "Current Ratio": {
            "current_assets": current_assets, "current_liabilities": current_liabilities,
            # Use the specific function, safe_calculate handles zero division
            "ratio_value": safe_calculate(calculate_current_ratio, current_assets, current_liabilities)
         },
         # ... other ratios ...
    }

    # ... rest of growth metrics, cash flow metrics, anomaly checks ...
    # (Code is the same as previous version)

    logging.info("Finished calculating financial ratios.")
    return ratios


# --- Red Flag Detection (Remains the same, uses data & ratios dicts) ---
def detect_financial_red_flags(data: Dict, ratios: Dict) -> Dict:
    # ... (keep the existing detect_financial_red_flags function as is) ...
    logging.info("Step 3: Detecting financial red flags (Python rules)...")
    # (Code is the same as previous version)
    red_flags = []
    # ... logic to append flags based on data and ratios ...
    logging.info(f"Detected {len(red_flags)} potential red flags via Python rules.")
    return {
        "red_flags": red_flags,
        "has_critical_issues": any(flag["severity"] == "High" for flag in red_flags),
        "has_concerns": len(red_flags) > 0
    }

# --- Updated LLM Generation Functions ---

async def generate_business_overview(client: genai.Client, extracted_data: Dict) -> str:
    """Generates a business overview using LLM, expecting JSON output."""
    logging.info("Step 4a: Generating business overview...")
    if not extracted_data:
        return "Error: Cannot generate overview without extracted data."

    # Prepare data for the prompt, ensure company_name exists
    company_name = extracted_data.get("company_name", "the company")
    prompt_data = {"extracted_data": json.dumps(extracted_data, indent=2), "company_name": company_name}
    prompt = OVERVIEW_PROMPT.format(**prompt_data)

    response_text = await generate_gemini_content(
        client=client, model_name=ANALYSIS_MODEL_NAME, prompt=prompt
    )
    parsed_response = parse_llm_json_response(response_text or "")

    if isinstance(parsed_response, dict) and "business_overview" in parsed_response:
        logging.info("Successfully generated and parsed business overview.")
        return parsed_response["business_overview"]
    else:
        logging.error(f"Failed to generate/parse business overview JSON. Raw response: {response_text[:500]}...")
        # Try to extract error if available
        error_msg = parsed_response.get("error", "Could not generate business overview.") if isinstance(parsed_response, dict) else "Could not generate business overview."
        return f"Error: {error_msg}"


async def generate_key_findings(client: genai.Client, extracted_data: Dict, calculated_ratios: Dict) -> Dict:
    """Generates key findings and LLM-identified red flags, expecting JSON."""
    logging.info("Step 4b: Generating key findings...")
    default_error_response = {
        "key_findings_analysis": {"executive_summary": "Analysis failed.", "profitability": "N/A", "liquidity_solvency": "N/A", "efficiency": "N/A", "trends_anomalies": "N/A"},
        "identified_red_flags": []
    }
    if not extracted_data or not calculated_ratios:
        logging.error("Cannot generate findings without extracted data and ratios.")
        return default_error_response

    # Add anomalies to extracted_data for context if they exist
    if "Anomalies" in calculated_ratios and calculated_ratios["Anomalies"]:
        extracted_data["calculated_anomalies"] = calculated_ratios["Anomalies"] # Use a distinct key

    prompt = FINDINGS_PROMPT.format(
        extracted_data=json.dumps(extracted_data, indent=2),
        calculated_ratios=json.dumps(calculated_ratios, indent=2)
    )
    response_text = await generate_gemini_content(
        client=client, model_name=ANALYSIS_MODEL_NAME, prompt=prompt
    )
    parsed_response = parse_llm_json_response(response_text or "")

    if isinstance(parsed_response, dict) and "key_findings_analysis" in parsed_response:
        logging.info("Successfully generated and parsed key findings.")
        # Validate structure slightly
        if "executive_summary" not in parsed_response["key_findings_analysis"]:
             logging.warning("Parsed findings JSON missing 'executive_summary'.")
             # Return structure with defaults where possible
             return {
                 "key_findings_analysis": parsed_response.get("key_findings_analysis", default_error_response["key_findings_analysis"]),
                 "identified_red_flags": parsed_response.get("identified_red_flags", [])
             }
        return parsed_response # Return the full parsed structure
    else:
        logging.error(f"Failed to generate/parse key findings JSON. Raw response: {response_text[:500]}...")
        error_msg = parsed_response.get("error", "Could not generate key findings analysis.") if isinstance(parsed_response, dict) else "Could not generate key findings analysis."
        # Return the default error structure but include the error message
        default_error_response["key_findings_analysis"]["executive_summary"] = f"Error: {error_msg}"
        return default_error_response


async def generate_sentiment_analysis(client: genai.Client, extracted_data: Dict) -> Dict:
    """Generates sentiment analysis, expecting JSON."""
    logging.info("Step 4c: Generating sentiment analysis...")
    default_error_response = {"sentiment_analysis": {"overall_tone": "Error", "key_indicators": "Analysis failed.", "forward_outlook_assessment": "N/A", "risk_disclosure_level": "N/A"}}
    notes_data = extracted_data.get("notes", {})
    if not notes_data:
        logging.warning("No 'notes' data found for sentiment analysis.")
        # Return a specific 'not applicable' structure
        return {"sentiment_analysis": {"overall_tone": "N/A", "key_indicators": "No notes/commentary provided in data.", "forward_outlook_assessment": "N/A", "risk_disclosure_level": "N/A"}}

    prompt = SENTIMENT_PROMPT.format(notes_data=json.dumps(notes_data, indent=2))
    response_text = await generate_gemini_content(
        client=client, model_name=ANALYSIS_MODEL_NAME, prompt=prompt
    )
    parsed_response = parse_llm_json_response(response_text or "")

    if isinstance(parsed_response, dict) and "sentiment_analysis" in parsed_response:
        logging.info("Successfully generated and parsed sentiment analysis.")
        return parsed_response # Return the full parsed structure
    else:
        logging.error(f"Failed to generate/parse sentiment analysis JSON. Raw response: {response_text[:500]}...")
        error_msg = parsed_response.get("error", "Could not generate sentiment analysis.") if isinstance(parsed_response, dict) else "Could not generate sentiment analysis."
        default_error_response["sentiment_analysis"]["key_indicators"] = f"Error: {error_msg}"
        return default_error_response


async def generate_business_model_ideas(client: genai.Client, extracted_data: Dict) -> Dict:
    """Generates business model ideas, expecting JSON."""
    logging.info("Step 4d: Generating business model ideas...")
    default_error_response = {"business_model_suggestions": [], "feasibility_notes": "Analysis failed."}
    if not extracted_data:
        logging.error("Cannot generate business model ideas without extracted data.")
        return default_error_response

    prompt = BUSINESS_MODEL_PROMPT.format(extracted_data=json.dumps(extracted_data, indent=2))
    response_text = await generate_gemini_content(
        client=client, model_name=ANALYSIS_MODEL_NAME, prompt=prompt
    )
    parsed_response = parse_llm_json_response(response_text or "")

    if isinstance(parsed_response, dict) and "business_model_suggestions" in parsed_response:
        logging.info("Successfully generated and parsed business model ideas.")
        return parsed_response # Return the full parsed structure
    else:
        logging.error(f"Failed to generate/parse business model ideas JSON. Raw response: {response_text[:500]}...")
        error_msg = parsed_response.get("error", "Could not generate business model suggestions.") if isinstance(parsed_response, dict) else "Could not generate business model suggestions."
        default_error_response["feasibility_notes"] = f"Error: {error_msg}"
        return default_error_response
