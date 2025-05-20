# financial_tools.py

def calculate_current_ratio(current_assets: float, current_liabilities: float) -> float:
    """Calculates the current ratio."""
    if current_liabilities == 0:
        return float('inf')  # Handle division by zero
    return current_assets / current_liabilities

def calculate_debt_to_equity_ratio(total_liabilities: float, shareholders_equity: float) -> float:
    """Calculates the debt-to-equity ratio."""
    if shareholders_equity == 0:
        return float('inf')
    return total_liabilities / shareholders_equity

# ... Add functions for ALL the required ratios from the document ...
def calculate_gross_margin_ratio(gross_profit: float, net_sales: float) -> float:
    if net_sales == 0:
        return 0.0
    return gross_profit / net_sales

def calculate_operating_margin_ratio(operating_income: float, net_sales: float) -> float:
    if net_sales == 0:
        return 0.0
    return operating_income / net_sales

def calculate_return_on_assets_ratio(net_income: float, total_assets: float) -> float:
    if total_assets == 0:
        return 0.0
    return net_income / total_assets

def calculate_return_on_equity_ratio(net_income: float, shareholders_equity: float) -> float:
    if shareholders_equity == 0:
        return 0.0
    return net_income / shareholders_equity

def calculate_asset_turnover_ratio(net_sales: float, average_total_assets: float) -> float:
    if average_total_assets == 0:
        return 0.0
    return net_sales / average_total_assets

def calculate_inventory_turnover_ratio(cost_of_goods_sold: float, average_inventory: float) -> float:
    if average_inventory == 0:
        return 0.0
    return cost_of_goods_sold / average_inventory

def calculate_receivables_turnover_ratio(net_credit_sales: float, average_accounts_receivable: float) -> float:
    if average_accounts_receivable == 0:
        return 0.0
    return net_credit_sales / average_accounts_receivable

def calculate_debt_ratio(total_liabilities: float, total_assets: float) -> float:
    if total_assets == 0:
        return 0.0
    return total_liabilities / total_assets

def calculate_interest_coverage_ratio(operating_income: float, interest_expenses: float) -> float:
    if interest_expenses == 0:
        return float('inf')
    return operating_income / interest_expenses

# --- ESG Analysis Tools (Future Enhancements) ---

# Placeholder for future ESG calculation functions.
# These functions would take extracted ESG data as input and derive scores,
# assess compliance, or calculate specific ESG metrics.

# Example: Calculate Carbon Intensity
# def calculate_carbon_intensity(ghg_emissions_scope1: float, ghg_emissions_scope2: float, revenue: float) -> Optional[float]:
#     """Calculates carbon intensity (e.g., Scope 1+2 emissions per unit of revenue)."""
#     if revenue is None or revenue == 0:
#         return None
#     if ghg_emissions_scope1 is None and ghg_emissions_scope2 is None:
#         return None
#     total_emissions = (ghg_emissions_scope1 or 0) + (ghg_emissions_scope2 or 0)
#     return total_emissions / revenue

# Example: Assess Board Diversity
# def assess_board_diversity(board_diversity_summary: Optional[str], board_independence_percentage: Optional[float]) -> Dict[str, Any]:
#     """Provides a qualitative or quantitative assessment of board diversity and independence."""
#     assessment = {
#         "diversity_notes": "No specific data provided" if not board_diversity_summary else board_diversity_summary,
#         "independence_level": "Unknown"
#     }
#     if board_independence_percentage is not None:
#         if board_independence_percentage > 0.5:
#             assessment["independence_level"] = "Majority Independent"
#         elif board_independence_percentage > 0.0 :
#             assessment["independence_level"] = "Minority Independent"
#         else:
#             assessment["independence_level"] = "Not Independent"
#     return assessment

# Example: Sentiment Analysis on ESG Text
# def analyze_esg_sentiment(text_data: Optional[str]) -> Optional[str]:
#     """Analyzes the sentiment of a given ESG qualitative text (e.g., using an NLTK model or another LLM call)."""
#     if not text_data:
#         return None
#     # Placeholder for actual sentiment analysis logic
#     # Could involve calling another LLM with a specific sentiment analysis prompt
#     if "positive progress" in text_data.lower():
#         return "Positive"
#     if "concerns" in text_data.lower() or "risks" in text_data.lower():
#         return "Negative"
#     return "Neutral"