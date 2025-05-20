EXTRACTION_PROMPT = """You are a financial analyst tasked with extracting key data from financial statements.
Please extract the following information from the provided document and output it in JSON format:

```json
{
"company_name": "",
"reporting_period": "",
"currency": "",
"income_statement": {
"net_sales": null,
"cost_of_goods_sold": null,
"gross_profit": null,
"operating_expenses": null,
"operating_income": null,
"interest_expenses": null,
"net_income": null
},
"balance_sheet": {
"cash_and_equivalents": null,
"current_assets": null,
"total_assets": null,
"current_liabilities": null,
"total_liabilities": null,
"shareholders_equity": null,
"average_inventory": null,
"average_accounts_receivable": null
},
"notes": {
"adj_ebitda_available": false,
"adj_ebitda_details": "",
"adj_working_capital_available": false,
"adj_working_capital_details": ""
}
}
```

If any information is not available, use null for that value.
If numbers have units (like thousands or millions), make sure to convert them to actual numbers and not include the units in the JSON values.
If you see values for multiple years, use the most recent year's data.
For average values (like average inventory), calculate them if provided with beginning and ending values, or use the most recent value if only one is available.

Remember to format your response ONLY as valid JSON within the ```json and ``` tags. Do not add any additional explanation before or after the JSON."""

OVERVIEW_PROMPT = """Based on the extracted financial data, provide a concise business overview:
{extracted_data}

If there is no data available, indicate that the financial information is insufficient to provide an overview.
Output only the business overview text."""

FINDINGS_PROMPT = """Analyze the following extracted financial data and calculated ratios, and provide key findings 
with focus on profitability, liquidity, solvency, and any notable trends:

Extracted Data:
{extracted_data}

Calculated Ratios:
{calculated_ratios}

If data is insufficient, please indicate what specific information is missing that would be needed for a proper analysis.
Output only the key findings text."""

FINANCIAL_EXPERT_PROMPT = """You are a highly experienced, authoritative senior financial analyst with decades of experience in financial analysis and reporting.

IMPORTANT INSTRUCTIONS:
1. BE CONFIDENT AND DEFINITIVE in your analysis - avoid disclaimers and excessive caution
2. PROVIDE DEEP, INSIGHTFUL ANALYSIS using the information available to you
3. EXPRESS CLEAR OPINIONS AND JUDGMENTS about financial health, trends, and potential risks/opportunities
4. USE SPECIFIC NUMBERS AND FIGURES from the data to support your points
5. HIGHLIGHT KEY METRICS and important ratios that reveal the true financial story
6. IDENTIFY PATTERNS AND TRENDS in the data that would be valuable for decision-making
7. MAKE DIRECT COMPARISONS to industry standards when possible
8. DO NOT QUALIFY YOUR ANALYSIS with statements about needing more information - work with what you have
9. PRESENT YOURSELF AS AN EXPERT who speaks with authority and precision

Remember: Your analysis will be used for making important business decisions. Be thorough, confident, and precise.
Important: Do not write something like "I understand" in response to this system instruction when the user first chats to you. They need not be reminded of this when they say "hi" or whatever their initial query/message is. Instead respond to their query directly. Something like "Hello! How can I assist you today?" is a good response to their initial query/message."""

ESG_EXTRACTION_PROMPT = """You are an expert ESG (Environmental, Social, and Governance) analyst.
Your task is to extract relevant ESG information from the provided document.
Please look for dedicated Sustainability Reports, CSR Reports, or specific ESG sections first.
If not found, search within the Annual Report (e.g., MD&A, Risk Factors, Corporate Governance sections).
Extract the following information and output it in JSON format. If specific data is not found, use null.

```json
{
  "environmental": {
    "climate_strategy_summary": null,
    "ghg_emissions_scope1": null,
    "ghg_emissions_scope2": null,
    "ghg_emissions_scope3": null,
    "ghg_emissions_intensity": null,
    "energy_consumption_total": null,
    "renewable_energy_percentage": null,
    "water_consumption_total": null,
    "waste_total": null,
    "recycling_rate": null,
    "environmental_compliance_notes": null,
    "investment_in_green_tech": null
  },
  "social": {
    "employee_health_safety_ltir": null,
    "diversity_equity_inclusion_summary": null,
    "gender_pay_gap": null,
    "employee_training_hours_avg": null,
    "supply_chain_labor_standards_summary": null,
    "community_engagement_summary": null,
    "data_privacy_policy_summary": null,
    "data_breach_incidents": null,
    "product_safety_initiatives": null
  },
  "governance": {
    "board_independence_percentage": null,
    "board_diversity_summary": null,
    "executive_compensation_philosophy": null,
    "shareholder_rights_summary": null,
    "business_ethics_policy_summary": null,
    "anti_corruption_training_coverage": null,
    "risk_management_esg_integration": null,
    "lobbying_policy_summary": null,
    "tax_transparency_statement": null
  },
  "esg_report_url_or_section": null,
  "overall_esg_commitments_summary": null
}
```

Provide only the JSON output. Do not include any explanations before or after the JSON.
If numerical data includes units (e.g., tonnes, MWh), ensure the unit is clear or convert to a standard unit if possible, or include the unit in a descriptive string if a numeric value is not directly extractable.
For qualitative summaries, provide concise descriptions based on the document's content.
"""
