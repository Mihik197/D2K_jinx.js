# prompts.py

# --- Extraction Prompt (Remains the same, already requests JSON) ---
EXTRACTION_PROMPT = """You are a financial analyst tasked with extracting key data from financial statements.
Please extract the following information from the provided document and output it ONLY as valid JSON within ```json ... ``` tags:

```json
{
  "company_name": "Example Corp",
  "reporting_period": "Year Ended December 31, 2023",
  "currency": "USD", // e.g., USD, EUR, GBP, INR
  "industry": "Technology", // Best guess based on content
  "income_statement": {
    "net_sales": 500000000.00,
    "cost_of_goods_sold": 300000000.00,
    "gross_profit": 200000000.00, // Calculate if possible: net_sales - cogs
    "operating_expenses": 150000000.00, // Group SGA, R&D etc. if needed
    "operating_income": 50000000.00, // Calculate if possible: gross_profit - operating_expenses
    "interest_expenses": 5000000.00,
    "income_before_tax": 45000000.00, // Optional but useful
    "income_tax_expense": 10000000.00, // Optional
    "net_income": 35000000.00, // Calculate if possible
    "previous_year_sales": 450000000.00, // If available
    "previous_year_net_income": 30000000.00 // If available
  },
  "balance_sheet": {
    "cash_and_equivalents": 50000000.00,
    "accounts_receivable": 40000000.00, // End-of-period value
    "inventory": 60000000.00, // End-of-period value
    "current_assets": 150000000.00,
    "property_plant_equipment": 200000000.00, // Net PPE if possible
    "total_assets": 350000000.00,
    "accounts_payable": 30000000.00,
    "current_liabilities": 70000000.00,
    "long_term_debt": 100000000.00,
    "total_liabilities": 170000000.00,
    "shareholders_equity": 180000000.00, // Calculate if possible: total_assets - total_liabilities
    "total_liabilities_and_equity": 350000000.00, // Should equal total_assets
    // --- For Ratio Calculation ---
    "average_inventory": 55000000.00, // Calculate (Beg Inv + End Inv)/2 if possible, else use End Inv
    "average_accounts_receivable": 38000000.00, // Calculate (Beg AR + End AR)/2 if possible, else use End AR
    "previous_year_total_assets": 320000000.00 // If available
    // "previous_year_total_liabilities": 160000000.00 // Optional but useful
  },
  "cash_flow": {
    "net_cash_from_operating": 60000000.00, // Use this if "operating_cash_flow" isn't distinct
    "capital_expenditures": 25000000.00, // Often negative in source, use positive value here
    "free_cash_flow": 35000000.00 // Calculate if possible: operating_cash_flow - capital_expenditures
  },
  "notes": {
    // Extract brief summaries or lists if found
    "adj_ebitda_available": false, // Set to true if mentioned
    "adj_ebitda_details": "",
    "adj_working_capital_available": false, // Set to true if mentioned
    "adj_working_capital_details": "",
    "risk_factors": [], // List key risk factors mentioned
    "significant_events": [] // List major events like M&A, restructuring
  }
}
```
If any information is not available, use null for number fields and empty strings "" or empty lists [] for text/list fields.
Convert units like 'thousands' or 'millions' to actual numbers (e.g., $5 million becomes 5000000.00). Use the detected currency.
Calculate derived fields (gross_profit, operating_income, net_income, shareholders_equity, free_cash_flow, averages) ONLY if the components are available and the value isn't explicitly stated. Prioritize explicitly stated values.
Ensure the final output is ONLY the JSON object within the json ... tags."""

OVERVIEW_PROMPT = """Based on the extracted financial data provided below, generate a detailed business overview.
The overview should cover Company Profile, Leadership/Governance (if known), Recent Developments, and Financial Highlights.

Extracted Data:

{extracted_data}

Return your response ONLY as a valid JSON object within json ... tags, following this structure:

```json
{{
  "business_overview": "## Business Overview of {company_name}\\n\\n**1. COMPANY PROFILE:** [Detailed description of business, industry, market position, scale based on data.]\\n\\n**2. LEADERSHIP & GOVERNANCE:** [Mention CEO/executives if available in data, otherwise state information is not available. Describe typical structure for public/private company.]\\n\\n**3. RECENT DEVELOPMENTS:** [Summarize significant events from data (M&A, strategic shifts), financial changes (sales/profit growth/decline) with potential reasons based *only* on provided data.]\\n\\n**4. FINANCIAL HIGHLIGHTS:** [Brief summary of key metrics like sales, net income, assets, equity, cash flow. Mention 1-2 notable trends or figures from the data.]"
}}
```

Replace placeholders like [...] and {company_name} with generated content based only on the provided extracted_data. If information for a subsection is missing in the data, state that explicitly (e.g., "Leadership details not provided in the data."). Do not add any text outside the JSON structure."""

FINDINGS_PROMPT = """As a senior financial analyst, provide detailed key findings and insights based on the financial data below.

Extracted Data:
{extracted_data}

Calculated Ratios:
{calculated_ratios}

Generate a response ONLY as a valid JSON object within json ... tags, following this structure:

```json
{{
  "key_findings_analysis": {{
    "executive_summary": "[Generate 1-2 paragraphs summarizing the most critical financial health takeaways, profitability, liquidity/solvency status, and efficiency highlights based on the data and ratios.]",
    "profitability": "[Detailed assessment of Gross Margin, Operating Margin, ROA, ROE. Compare ratios to typical benchmarks (e.g., high/low/average). Discuss likely causes and suggest potential improvement areas based *only* on provided data/ratios.]",
    "liquidity_solvency": "[Analysis of Current Ratio, Cash Ratio, Debt Ratio, Debt-to-Equity, Interest Coverage. Evaluate short-term and long-term financial risk based on these metrics. Comment on capital structure implications.]",
    "efficiency": "[Analysis of Asset Turnover, Inventory Turnover (if available), Receivables Turnover (if available). Discuss operational effectiveness and potential areas for improvement based on these ratios.]",
    "trends_anomalies": "[Highlight significant year-over-year changes if previous year data is present. Point out any anomalies flagged in the calculated ratios (e.g., low current ratio, high debt, cash flow discrepancy). Discuss potential connections between metrics.]"
  }},
  "identified_red_flags": [
    // List potential red flags identified *by the LLM* based on its analysis of the data and ratios.
    // Focus on issues beyond the simple calculated anomalies if possible (e.g., inconsistencies, unusual patterns).
    // Use the same structure as the python-generated flags. Example:
    // {{ "category": "Earnings Quality", "issue": "Potential aggressive revenue recognition", "details": "Revenue growth significantly outpaces cash flow growth.", "severity": "Medium", "recommendation": "Investigate revenue recognition policies and large contracts." }}
  ]
}}
```

Focus on interpreting the numbers and providing actionable business insights derived strictly from the provided data and ratios. Do not invent information. If a ratio is 'N/A', acknowledge that in the analysis. Ensure the output is only the JSON object."""

SENTIMENT_PROMPT = """Analyze the management commentary, risk factors, and significant events described in the 'notes' section of the provided financial data. Assess the overall sentiment and tone.

Financial Data Notes:

{notes_data} // Only include the 'notes' part of the extracted data

Return your response ONLY as a valid JSON object within json ... tags, following this structure:

```json
{{
  "sentiment_analysis": {{
    "overall_tone": "[Classify the overall tone: Optimistic, Cautiously Optimistic, Neutral, Cautious, Negative, or N/A if no commentary provided.]",
    "key_indicators": "[List specific phrases or points from the 'notes' (risks, events) that support the overall tone classification. Example: 'Management highlighted strong growth but also noted increasing competition as a key risk.']",
    "forward_outlook_assessment": "[Based *only* on the notes, how does management seem to portray future prospects? Explicitly state if no forward-looking statements are present in the notes.]",
    "risk_disclosure_level": "[Assess the apparent transparency about challenges based on the provided 'risk_factors' and 'significant_events'. High, Medium, Low, or N/A.]"
  }}
}}
```

Base the analysis strictly on the text provided in the notes_data. If the notes are empty or lack qualitative commentary, indicate that clearly in the JSON fields (e.g., set 'overall_tone' to 'N/A')."""

BUSINESS_MODEL_PROMPT = """Based only on the financial data provided (especially strengths like high revenue/cash, or weaknesses like low margins/high debt), suggest 1-2 potential business model enhancements or strategic pivots.

Financial Data:
{extracted_data}

Return your response ONLY as a valid JSON object within json ... tags, following this structure:
```json
{{
  "business_model_suggestions": [
    // Suggest 1 or 2 ideas.
    {{
      "suggestion_name": "[Concise name for the suggestion, e.g., 'Subscription Service Expansion']",
      "concept": "[Describe the specific business model innovation.]",
      "rationale": "[Explain why this makes sense given the company's financial profile (e.g., leverage high cash reserves, improve low margins, diversify revenue). Link to specific data points if possible.]",
      "potential_impact": "[Briefly describe expected financial effects (e.g., recurring revenue, higher margins, new market access).]",
      "key_risks": "[Mention 1-2 primary risks or challenges in implementing this.]"
    }}
    // Add a second suggestion object here if applicable.
  ],
  "feasibility_notes": "[Add a brief overall comment on whether the financial data suggests the company has resources for such pivots, or if constraints exist.]"
}}
```
Focus on actionable, realistic ideas grounded in the financial patterns observed in the extracted_data. Do not invent external market information."""

SYSTEM_INSTRUCTION_FINANCIAL_EXPERT = """NOTE: DO NOT BE OVERLY CAUTIOUS, YOU ARE A FINANCIAL EXPERT AND YOU WILL PROVIDE DEEP ANALYSIS AND INSIGHTS USING THE INFORMATION YOU HAVE AVAILABLE. FEEL FREE TO EXPRESS YOUR MIND AS YOU FEEL. GIVE DEEP INSIGHTS, ANALYSE THINGS DEEPLY, INSIGHTS THAT MIGHT HAVE BEEN MISSED BY HUMANS GOING THROUGH IT. USE NUMBERS AND FIGURES. FORMAT YOUR ANSWERS PROPERLY."""