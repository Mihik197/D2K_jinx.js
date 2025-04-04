# financial_tools.py

def calculate_current_ratio(current_assets, current_liabilities):
    """Calculates the Current Ratio."""
    if current_liabilities == 0:
        return None  # Avoid division by zero
    return current_assets / current_liabilities

def calculate_debt_to_equity_ratio(total_liabilities, shareholders_equity):
    """Calculates the Debt to Equity Ratio."""
    if shareholders_equity == 0:
        return None
    return total_liabilities / shareholders_equity

def calculate_gross_margin_ratio(gross_profit, net_sales):
    """Calculates the Gross Margin Ratio."""
    if net_sales == 0:
        return None
    return gross_profit / net_sales

def calculate_operating_margin_ratio(operating_income, net_sales):
    """Calculates the Operating Margin Ratio."""
    if net_sales == 0:
        return None
    return operating_income / net_sales

def calculate_return_on_assets_ratio(net_income, total_assets):
    """Calculates the Return on Assets (ROA) Ratio."""
    if total_assets == 0:
        return None
    return net_income / total_assets

def calculate_return_on_equity_ratio(net_income, shareholders_equity):
    """Calculates the Return on Equity (ROE) Ratio."""
    if shareholders_equity == 0:
        return None
    return net_income / shareholders_equity

def calculate_asset_turnover_ratio(net_sales, average_total_assets):
    """Calculates the Asset Turnover Ratio."""
    if average_total_assets == 0:
        return None
    return net_sales / average_total_assets

def calculate_inventory_turnover_ratio(cost_of_goods_sold, average_inventory):
    """Calculates the Inventory Turnover Ratio."""
    if average_inventory == 0:
        return None
    return cost_of_goods_sold / average_inventory

def calculate_receivables_turnover_ratio(net_credit_sales, average_accounts_receivable):
    """Calculates the Receivables Turnover Ratio."""
    if average_accounts_receivable == 0:
        return None
    return net_credit_sales / average_accounts_receivable

def calculate_debt_ratio(total_liabilities, total_assets):
    """Calculates the Debt Ratio."""
    if total_assets == 0:
        return None
    return total_liabilities / total_assets

def calculate_interest_coverage_ratio(operating_income, interest_expenses):
    """Calculates the Interest Coverage Ratio."""
    # Handle case where interest expense is zero or negative (unusual but possible)
    if interest_expenses <= 0:
        return float('inf') if operating_income > 0 else 0 # Or None, depending on interpretation
    return operating_income / interest_expenses

def safe_calculate(func, numerator, denominator, default="N/A"):
    """Helper to safely call calculation functions."""
    try:
        # Ensure numeric inputs where expected
        num = float(numerator) if numerator is not None else 0
        den = float(denominator) if denominator is not None else 0

        if den == 0:
            # Special handling for interest coverage where 0 interest might mean infinite coverage
            if func == calculate_interest_coverage_ratio and num > 0:
                return float('inf')
            return default

        result = func(num, den)
        return result if result is not None else default
    except (TypeError, ValueError, ZeroDivisionError):
        return default
    except Exception:
        return default # Catch any other unexpected errors