from reportlab.lib.pagesizes import LETTER, A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.platypus.flowables import Flowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.graphics.shapes import Drawing, Line
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.linecharts import HorizontalLineChart
import json
import io
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments

def format_esg_key(key_name: str) -> str:
    """Converts a snake_case string to Title Case."""
    if not key_name:
        return ""
    return " ".join(word.capitalize() for word in key_name.split("_"))

# Custom flowable for page headers and footers
class HeaderFooter(Flowable):
    def __init__(self, width, height, report_info):
        Flowable.__init__(self)
        self.width = width
        self.height = height
        self.report_info = report_info
    
    def draw(self):
        # Save canvas state
        self.canv.saveState()
        
        # Draw header
        self.canv.setFont("Helvetica-Bold", 10)
        self.canv.setFillColor(colors.darkblue)
        self.canv.drawString(0.5*inch, self.height - 0.5*inch, "FINANCIAL ANALYSIS REPORT")
        
        # Draw company name if available
        company_name = self.report_info.get("company_name", "")
        if company_name:
            self.canv.setFont("Helvetica", 8)
            self.canv.drawRightString(self.width - 0.5*inch, self.height - 0.5*inch, company_name)
        
        # Draw horizontal line under header
        self.canv.setStrokeColor(colors.grey)
        self.canv.setLineWidth(0.5)
        self.canv.line(0.5*inch, self.height - 0.6*inch, self.width - 0.5*inch, self.height - 0.6*inch)
        
        # Draw footer with page number
        self.canv.setFont("Helvetica", 8)
        self.canv.setFillColor(colors.black)
        self.canv.drawString(0.5*inch, 0.4*inch, f"Generated on: {datetime.now().strftime('%B %d, %Y')}")
        self.canv.drawRightString(self.width - 0.5*inch, 0.4*inch, "Page ${page_number}")
        
        # Draw horizontal line above footer
        self.canv.line(0.5*inch, 0.5*inch, self.width - 0.5*inch, 0.5*inch)
        
        # Restore canvas state
        self.canv.restoreState()

# Custom page template class for headers and footers
class PageTemplate(Flowable):
    def __init__(self, doc):
        super().__init__()
        self.doc = doc
        self.page_width, self.page_height = doc.pagesize
        
    def draw(self):
        page_num = self.doc.page
        self.canv.saveState()
        self.canv.setFont('Helvetica', 9)
        self.canv.drawRightString(self.page_width - 0.75*inch, 0.75*inch, f"Page {page_num}")
        self.canv.restoreState()

class HorizontalRule(Flowable):
    """A custom Flowable for drawing a horizontal line"""
    def __init__(self, width, thickness=1, color=colors.black, spacer=0.1):
        Flowable.__init__(self)
        self.width = width
        self.thickness = thickness
        self.color = color
        self.spacer = spacer
        
    def __repr__(self):
        return "HorizontalRule(w=%s)" % self.width
        
    def draw(self):
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, 0, self.width, 0)

def format_financial_value(value):
    """Format a financial value with appropriate presentation"""
    if value is None:
        return "N/A"
    elif isinstance(value, (int, float)):
        if value >= 1000000:
            return f"₹{value/1000000:.2f}M"
        elif value >= 1000:
            return f"₹{value/1000:.2f}K"
        else:
            return f"₹{value:.2f}"
    else:
        return str(value)

def create_pie_chart(data, title):
    """Create a pie chart for the given data"""
    # Filter data to include only positive values
    filtered_data = {k: v for k, v in data.items() if isinstance(v, (int, float)) and v > 0}
    if not filtered_data:
        return None
        
    # Create pie chart
    plt.figure(figsize=(7, 5))
    plt.pie(
        filtered_data.values(), 
        labels=filtered_data.keys(),
        autopct='%1.1f%%', 
        startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1},
        shadow=True,
        textprops={'fontsize': 9}
    )
    plt.axis('equal')  
    plt.title(title, fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # Save chart to bytes
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png', dpi=150, bbox_inches='tight')
    img_data.seek(0)
    plt.close()
    
    return Image(img_data, width=6*inch, height=4*inch)

def create_bar_chart(data, title, colors_list=None):
    """Create a bar chart for the given data"""
    if not data:
        return None
    
    # Prepare data
    labels = list(data.keys())
    values = list(data.values())
    
    # Create color gradient if not provided
    if colors_list is None:
        colors_list = plt.cm.Blues(np.linspace(0.4, 0.8, len(values)))
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors_list, width=0.6)
    
    # Add value labels above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.02 * max(values),
            f'{height:.4f}',
            ha='center', va='bottom',
            fontsize=9
        )
    
    plt.title(title, fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save chart to bytes
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png', dpi=150, bbox_inches='tight')
    img_data.seek(0)
    plt.close()
    
    return Image(img_data, width=6.5*inch, height=4*inch)

def generate_pdf_report(report_data: dict, output_path: str):
    """
    Generate a visually appealing PDF report based on provided report data.

    report_data should be a dict such as:
    {
       "business_overview": "string",
       "key_findings": "string",
       "extracted_data": { ... },
       "calculated_ratios": { ... }
    }
    """
    # Create PDF document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=LETTER,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=1*inch,
        bottomMargin=1*inch
    )
    
    # Extract company info for headers
    extracted_data = report_data.get("extracted_data", {})
    company_name = extracted_data.get("company_name", "")
    reporting_period = extracted_data.get("reporting_period", "")
    
    # Prepare styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=20,
        fontName='Helvetica-Bold',
        alignment=1,  # Center alignment
        textColor=colors.darkblue,
        spaceAfter=12
    )
    
    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=16,
        fontName='Helvetica-Bold',
        textColor=colors.darkblue,
        spaceBefore=12,
        spaceAfter=6,
        borderPadding=5,
        borderWidth=0,
        borderRadius=5,
        borderColor=colors.lightgrey
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=14,
        fontName='Helvetica-Bold',
        textColor=colors.darkslategray,
        spaceBefore=10,
        spaceAfter=4
    )
    
    normal_style = styles["Normal"]
    
    # Create custom styles
    body_style = ParagraphStyle(
        'BodyText',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        spaceBefore=6,
        spaceAfter=6
    )
    
    # Add a highlighted info style
    info_style = ParagraphStyle(
        'InfoText',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        spaceBefore=6,
        spaceAfter=6,
        backColor=colors.lavender,
        borderWidth=1,
        borderColor=colors.lightblue,
        borderPadding=10,
        borderRadius=5
    )
    
    # Add a key metrics style
    metric_style = ParagraphStyle(
        'MetricText',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.darkslateblue,
        backColor=colors.lightgrey,
        borderWidth=1,
        borderColor=colors.grey,
        borderPadding=5,
        borderRadius=5,
        alignment=1  # Center alignment
    )
    
    # Create story content
    story = []
    
    # Add title page
    story.append(Paragraph("Financial Analysis Report", title_style))
    if company_name:
        story.append(Paragraph(f"<b>{company_name}</b>", ParagraphStyle(
            'CompanyName',
            parent=styles['Normal'],
            fontSize=16,
            alignment=1,
            spaceAfter=6
        )))
    if reporting_period:
        story.append(Paragraph(f"Reporting Period: {reporting_period}", ParagraphStyle(
            'ReportingPeriod',
            parent=styles['Normal'],
            fontSize=12,
            alignment=1
        )))
    
    story.append(Spacer(1, 0.5 * inch))
    
    # Add date and disclaimer
    current_date = datetime.now().strftime("%B %d, %Y")
    story.append(Paragraph(f"Generated on: {current_date}", ParagraphStyle(
        'DateText',
        parent=styles['Normal'],
        fontSize=10,
        alignment=1,
        textColor=colors.gray
    )))
    
    story.append(Spacer(1, 2 * inch))
    
    # Add a decorative line before the disclaimer
    story.append(HorizontalRule(450, thickness=1, color=colors.lightgrey))
    story.append(Spacer(1, 0.2 * inch))
    
    disclaimer_text = "This report contains an analysis of financial statements and has been auto-generated using AI. " \
                      "The information presented should be verified with original financial documents before making " \
                      "any business decisions."
    
    story.append(Paragraph(disclaimer_text, ParagraphStyle(
        'DisclaimerText',
        parent=styles['Italic'],
        fontSize=8,
        alignment=1,
        textColor=colors.gray
    )))
    
    # Add page break after title page
    story.append(PageBreak())
    
    # Add table of contents header
    story.append(Paragraph("TABLE OF CONTENTS", ParagraphStyle(
        'TOCHeader',
        parent=styles['Heading1'],
        fontSize=14,
        alignment=1,
        spaceAfter=20
    )))
    
    # Add simple table of contents
    toc_items = [
        ("1. Business Overview", "3"),
        ("2. Key Findings", "4"),
        ("3. Financial Ratios", "5"),
        ("   3.1 Profitability Ratios", "5"),
        ("   3.2 Liquidity Ratios", "6"),
        ("   3.3 Solvency Ratios", "7"),
        ("   3.4 Efficiency Ratios", "8"),
        ("4. Financial Statements", "9"),
        ("   4.1 Income Statement", "9"),
        ("   4.2 Balance Sheet", "10"),
        ("5. Additional Analysis", "11"), # Assuming notes section exists
        ("6. ESG Insights", "12") # Placeholder page, adjust if dynamic TOC is implemented
    ]
    
    toc_data = [[item, page] for item, page in toc_items]
    toc_table = Table(toc_data, colWidths=[5*inch, 0.5*inch])
    toc_table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('LINEABOVE', (0, -1), (-1, -1), 1, colors.grey),
        ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.whitesmoke])
    ]))
    
    story.append(toc_table)
    
    # Add page break after table of contents
    story.append(PageBreak())
    
    # Add Business Overview section
    story.append(Paragraph("1. BUSINESS OVERVIEW", heading1_style))
    story.append(HorizontalRule(450, thickness=2, color=colors.lightsteelblue))
    story.append(Spacer(1, 0.15 * inch))
    
    overview_text = report_data.get("business_overview", "No overview available.")
    story.append(Paragraph(overview_text, body_style))
    story.append(Spacer(1, 0.3 * inch))
    
    # Add Key Metrics summary box if data is available
    if extracted_data:
        # Create a summary of key metrics in a visually appealing box
        income_statement = extracted_data.get("income_statement", {})
        balance_sheet = extracted_data.get("balance_sheet", {})
        
        if income_statement or balance_sheet:
            story.append(Paragraph("Key Metrics", heading2_style))
            
            # Format key metrics data
            key_metrics = []
            
            if income_statement.get("net_sales"):
                key_metrics.append(["Revenue", format_financial_value(income_statement.get("net_sales"))])
            
            if income_statement.get("net_income"):
                key_metrics.append(["Net Income", format_financial_value(income_statement.get("net_income"))])
                
            if balance_sheet.get("total_assets"):
                key_metrics.append(["Total Assets", format_financial_value(balance_sheet.get("total_assets"))])
            
            if balance_sheet.get("shareholders_equity"):
                key_metrics.append(["Shareholder's Equity", format_financial_value(balance_sheet.get("shareholders_equity"))])
            
            if key_metrics:
                # Create two-column layout for key metrics
                metrics_data = [key_metrics[i:i+2] for i in range(0, len(key_metrics), 2)]
                for row in metrics_data:
                    if len(row) < 2:
                        row.append(["", ""])  # Add empty cell for odd number of metrics
                
                metric_items = []
                for metric_row in metrics_data:
                    row_items = []
                    for label, value in metric_row:
                        # Create style for each metric box
                        metric_box = Table([[Paragraph(f"<b>{label}</b>", body_style)], 
                                           [Paragraph(value, ParagraphStyle('MetricValue', 
                                                                         parent=body_style, 
                                                                         fontSize=12, 
                                                                         alignment=1))]], 
                                          colWidths=[2.5*inch], 
                                          rowHeights=[0.3*inch, 0.5*inch])
                        metric_box.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.lightsteelblue),
                            ('BACKGROUND', (0, 1), (-1, 1), colors.white),
                            ('BOX', (0, 0), (-1, -1), 1, colors.lightgrey),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ]))
                        row_items.append(metric_box)
                    
                    # Create row with metrics
                    metric_row = Table([row_items], colWidths=[2.5*inch, 2.5*inch])
                    metric_row.setStyle(TableStyle([
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ]))
                    metric_items.append(metric_row)
                
                # Add all metrics to the story
                for item in metric_items:
                    story.append(item)
                    story.append(Spacer(1, 0.1 * inch))
    
    story.append(PageBreak())
    
    # Add Key Findings section
    story.append(Paragraph("2. KEY FINDINGS", heading1_style))
    story.append(HorizontalRule(450, thickness=2, color=colors.lightsteelblue))
    story.append(Spacer(1, 0.15 * inch))
    
    findings_text = report_data.get("key_findings", "No findings available.")
    
    # Format the findings text better with bullet points
    formatted_findings = ""
    if "**" in findings_text:  # Convert markdown-like formatting to proper bullets
        for line in findings_text.split("\n"):
            if "**" in line:
                parts = line.split("**")
                if len(parts) >= 3:  # Should have content before, during, and after **
                    category, content = parts[1], parts[2]
                    formatted_findings += f"<strong>{category}</strong>: {content}<br/><br/>"
            else:
                formatted_findings += line + "<br/>"
    else:
        # If no markdown formatting found, try to organize by paragraphs
        paragraphs = findings_text.split("\n")
        for para in paragraphs:
            if para.strip():
                if ":" in para:
                    # This looks like it might be a category: description format
                    parts = para.split(":", 1)
                    formatted_findings += f"<strong>{parts[0]}</strong>: {parts[1]}<br/><br/>"
                else:
                    formatted_findings += para + "<br/><br/>"
    
    # Use formatted findings if we have them, otherwise use the original text
    if formatted_findings:
        story.append(Paragraph(formatted_findings, info_style))
    else:
        story.append(Paragraph(findings_text, info_style))
    
    story.append(PageBreak())
    
    # Add Financial Ratios section with charts
    ratios = report_data.get("calculated_ratios", {})
    if ratios:
        story.append(Paragraph("3. FINANCIAL RATIOS", heading1_style))
        story.append(HorizontalRule(450, thickness=2, color=colors.lightsteelblue))
        story.append(Spacer(1, 0.15 * inch))
        
        # Add brief explanation of financial ratios
        ratio_explanation = "Financial ratios are tools used to analyze a company's financial performance and condition. " \
                           "They are calculated from data in the company's financial statements and provide insights " \
                           "into profitability, operational efficiency, liquidity, and solvency."
        story.append(Paragraph(ratio_explanation, body_style))
        story.append(Spacer(1, 0.2 * inch))
        
        # Group ratios by category for better visualization
        ratio_categories = {
            "Profitability": ["Gross Margin Ratio", "Operating Margin Ratio", "Return on Assets Ratio", "Return on Equity Ratio"],
            "Liquidity": ["Current Ratio", "Cash Ratio"],
            "Solvency": ["Debt to Equity Ratio", "Debt Ratio", "Interest Coverage Ratio"],
            "Efficiency": ["Asset Turnover Ratio", "Inventory Turnover Ratio", "Receivables Turnover Ratio"]
        }
        
        # Add interpretation guidelines for each ratio category
        category_explanations = {
            "Profitability": "Profitability ratios indicate how well a company generates profit relative to its revenue, assets, and equity.",
            "Liquidity": "Liquidity ratios measure a company's ability to pay off its short-term debt obligations.",
            "Solvency": "Solvency ratios evaluate a company's ability to meet its long-term obligations.",
            "Efficiency": "Efficiency ratios gauge how well a company utilizes its assets and resources."
        }
        
        # Create section number counter
        section_counter = 1
        
        # Create ratio tables and charts by category
        for category, ratio_names in ratio_categories.items():
            # Filter ratios that exist in our data
            category_ratios = {name: ratios.get(name, {}).get("ratio_value", "N/A") 
                              for name in ratio_names if name in ratios}
            
            if category_ratios:
                # Add subsection heading
                story.append(Paragraph(f"3.{section_counter} {category} Ratios", heading2_style))
                section_counter += 1
                
                # Add category explanation
                if category in category_explanations:
                    story.append(Paragraph(category_explanations[category], body_style))
                    story.append(Spacer(1, 0.1 * inch))
                
                # Create table data
                table_data = [["Ratio", "Value", "Interpretation"]]
                
                # Add interpretation for each ratio
                interpretations = {
                    "Gross Margin Ratio": lambda x: "Excellent" if x > 0.5 else "Good" if x > 0.3 else "Average" if x > 0.2 else "Below Average",
                    "Operating Margin Ratio": lambda x: "Excellent" if x > 0.15 else "Good" if x > 0.1 else "Average" if x > 0.05 else "Below Average",
                    "Return on Assets Ratio": lambda x: "Excellent" if x > 0.1 else "Good" if x > 0.06 else "Average" if x > 0.03 else "Below Average",
                    "Return on Equity Ratio": lambda x: "Excellent" if x > 0.2 else "Good" if x > 0.15 else "Average" if x > 0.1 else "Below Average",
                    "Current Ratio": lambda x: "Excellent" if 1.5 <= x <= 3 else "Good" if 1.2 <= x < 1.5 or 3 < x <= 4 else "Average" if 1 <= x < 1.2 else "Below Average",
                    "Cash Ratio": lambda x: "Excellent" if x > 0.5 else "Good" if x > 0.3 else "Average" if x > 0.1 else "Below Average",
                    "Debt to Equity Ratio": lambda x: "Excellent" if 0 <= x <= 1 else "Good" if 1 < x <= 1.5 else "Average" if 1.5 < x <= 2 else "Below Average",
                    "Debt Ratio": lambda x: "Excellent" if 0 <= x <= 0.3 else "Good" if 0.3 < x <= 0.5 else "Average" if 0.5 < x <= 0.6 else "Below Average",
                    "Interest Coverage Ratio": lambda x: "Excellent" if x > 5 else "Good" if x > 3 else "Average" if x > 1.5 else "Below Average",
                    "Asset Turnover Ratio": lambda x: "Excellent" if x > 2 else "Good" if x > 1 else "Average" if x > 0.5 else "Below Average",
                    "Inventory Turnover Ratio": lambda x: "Excellent" if x > 10 else "Good" if x > 6 else "Average" if x > 3 else "Below Average",
                    "Receivables Turnover Ratio": lambda x: "Excellent" if x > 10 else "Good" if x > 8 else "Average" if x > 4 else "Below Average"
                }
                
                for ratio_name, ratio_value in category_ratios.items():
                    if isinstance(ratio_value, (int, float)):
                        # Format the value with 4 decimal places
                        formatted_value = f"{ratio_value:.4f}"
                        
                        # Get interpretation
                        if ratio_name in interpretations:
                            interpretation = interpretations[ratio_name](ratio_value)
                        else:
                            interpretation = "N/A"
                    else:
                        formatted_value = str(ratio_value)
                        interpretation = "N/A"
                    
                    table_data.append([ratio_name, formatted_value, interpretation])
                
                # Create a styled table
                ratio_table = Table(table_data, colWidths=[3*inch, 1*inch, 1.5*inch])
                ratio_table.setStyle(TableStyle([
                    # Header styling
                    ('BACKGROUND', (0, 0), (-1, 0), colors.cornflowerblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                    # Data styling
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
                    ('ALIGN', (2, 1), (2, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
                    # Conditional formatting for interpretations
                    ('TEXTCOLOR', (2, 1), (2, -1), colors.green, lambda x, y: y >= 1 and table_data[y][2] == "Excellent"),
                    ('TEXTCOLOR', (2, 1), (2, -1), colors.forestgreen, lambda x, y: y >= 1 and table_data[y][2] == "Good"),
                    ('TEXTCOLOR', (2, 1), (2, -1), colors.orange, lambda x, y: y >= 1 and table_data[y][2] == "Average"),
                    ('TEXTCOLOR', (2, 1), (2, -1), colors.red, lambda x, y: y >= 1 and table_data[y][2] == "Below Average")
                ]))
                story.append(ratio_table)
                story.append(Spacer(1, 0.2 * inch))
                
                # Create visualization for this category
                chart_data = {k: v for k, v in category_ratios.items() if isinstance(v, (int, float))}
                if chart_data:
                    # Choose appropriate chart colors based on the category
                    if category == "Profitability":
                        colors_list = plt.cm.Greens(np.linspace(0.4, 0.8, len(chart_data)))
                    elif category == "Liquidity":
                        colors_list = plt.cm.Blues(np.linspace(0.4, 0.8, len(chart_data)))
                    elif category == "Solvency":
                        colors_list = plt.cm.Oranges(np.linspace(0.4, 0.8, len(chart_data)))
                    else:  # Efficiency
                        colors_list = plt.cm.Purples(np.linspace(0.4, 0.8, len(chart_data)))
                        
                    chart = create_bar_chart(chart_data, f"{category} Ratios", colors_list)
                    if chart:
                        story.append(chart)
                        story.append(Spacer(1, 0.2 * inch))
                
                # Add page break between ratio categories
                story.append(PageBreak())
        
        # Add radar chart for comprehensive ratio visualization
        profitability_ratios = {k: ratios.get(k, {}).get("ratio_value", 0) 
                              for k in ["Gross Margin Ratio", "Operating Margin Ratio", 
                                       "Return on Assets Ratio", "Return on Equity Ratio"] 
                              if k in ratios and isinstance(ratios.get(k, {}).get("ratio_value"), (int, float))}
        
        liquidity_solvency = {k: ratios.get(k, {}).get("ratio_value", 0) 
                            for k in ["Current Ratio", "Cash Ratio", "Debt to Equity Ratio", "Debt Ratio"] 
                            if k in ratios and isinstance(ratios.get(k, {}).get("ratio_value"), (int, float))}
        
        efficiency_ratios = {k: ratios.get(k, {}).get("ratio_value", 0) 
                           for k in ["Asset Turnover Ratio", "Inventory Turnover Ratio", "Receivables Turnover Ratio"] 
                           if k in ratios and isinstance(ratios.get(k, {}).get("ratio_value"), (int, float))}
        
        # Create pie chart showing the breakdown of revenue and expenses
        if extracted_data.get("income_statement"):
            income_data = extracted_data.get("income_statement", {})
            if income_data.get("net_sales") and income_data.get("cost_of_goods_sold") and income_data.get("operating_expenses"):
                expense_data = {
                    "Cost of Goods Sold": income_data.get("cost_of_goods_sold", 0),
                    "Operating Expenses": income_data.get("operating_expenses", 0),
                    "Net Income": income_data.get("net_income", 0)
                }
                
                pie_chart = create_pie_chart(expense_data, "Revenue Breakdown")
                if pie_chart:
                    story.append(Paragraph("Revenue & Expense Breakdown", heading2_style))
                    story.append(pie_chart)
                    story.append(Spacer(1, 0.2 * inch))
    
    # Add Financial Statements section
    story.append(Paragraph("4. FINANCIAL STATEMENTS", heading1_style))
    story.append(HorizontalRule(450, thickness=2, color=colors.lightsteelblue))
    story.append(Spacer(1, 0.15 * inch))
    
    # Add brief explanation
    story.append(Paragraph("The following sections contain the extracted financial data from the company's statements.", body_style))
    story.append(Spacer(1, 0.2 * inch))
    
    # Income Statement Section
    if "income_statement" in extracted_data and extracted_data["income_statement"]:
        story.append(Paragraph("4.1 Income Statement", heading2_style))
        
        income_data = extracted_data["income_statement"]
        income_table_data = [["Item", "Value (in thousands)"]]
        
        # Add description of income statement
        income_desc = "The Income Statement shows the company's revenues, expenses, and profits over a period of time."
        story.append(Paragraph(income_desc, body_style))
        story.append(Spacer(1, 0.15 * inch))
        
        # Format the income statement data with improved organization and formatting
        key_order = [
            "net_sales", "cost_of_goods_sold", "gross_profit", 
            "operating_expenses", "operating_income", 
            "interest_expenses", "net_income"
        ]
        
        # Add all items that exist in our order
        for key in key_order:
            if key in income_data:
                # Format the key to be more readable
                formatted_key = " ".join(word.capitalize() for word in key.split("_"))
                value = income_data[key]
                
                # Format the value
                if value is None:
                    formatted_value = "N/A"
                else:
                    formatted_value = f"{value:,.2f}"
                    
                income_table_data.append([formatted_key, formatted_value])
        
        # Add any remaining items not in our key_order
        for key, value in income_data.items():
            if key not in key_order:
                # Format the key to be more readable
                formatted_key = " ".join(word.capitalize() for word in key.split("_"))
                
                # Format the value
                if value is None:
                    formatted_value = "N/A"
                else:
                    formatted_value = f"{value:,.2f}"
                    
                income_table_data.append([formatted_key, formatted_value])
        
        # Style the income statement table
        table = Table(income_table_data, colWidths=[3*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.cornflowerblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
            # Highlight key rows
            ('BACKGROUND', (0, 1), (-1, 1), colors.lightgrey),  # Net Sales
            ('BACKGROUND', (0, 3), (-1, 3), colors.lightgrey),  # Gross Profit
            ('BACKGROUND', (0, 5), (-1, 5), colors.lightgrey),  # Operating Income
            ('BACKGROUND', (0, -1), (-1, -1), colors.lightblue),  # Net Income
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),  # Make Net Income bold
        ]))
        story.append(table)
        story.append(Spacer(1, 0.3 * inch))
    
    # Balance Sheet Section
    if "balance_sheet" in extracted_data and extracted_data["balance_sheet"]:
        story.append(Paragraph("4.2 Balance Sheet", heading2_style))
        
        balance_data = extracted_data["balance_sheet"]
        
        # Add description of balance sheet
        balance_desc = "The Balance Sheet presents the company's financial position at a point in time, showing assets, liabilities, and shareholders' equity."
        story.append(Paragraph(balance_desc, body_style))
        story.append(Spacer(1, 0.15 * inch))
        
        # Group balance sheet items by category
        assets_items = [
            ("cash_and_equivalents", "Cash and Equivalents"),
            ("current_assets", "Current Assets"),
            ("average_inventory", "Average Inventory"),
            ("average_accounts_receivable", "Average Accounts Receivable"),
            ("total_assets", "Total Assets")
        ]
        
        liabilities_equity_items = [
            ("current_liabilities", "Current Liabilities"),
            ("total_liabilities", "Total Liabilities"),
            ("shareholders_equity", "Shareholders' Equity")
        ]
        
        # Create assets table
        assets_table_data = [["Assets", "Value (in thousands)"]]
        
        for key, label in assets_items:
            if key in balance_data:
                value = balance_data[key]
                
                # Format the value
                if value is None:
                    formatted_value = "N/A"
                else:
                    formatted_value = f"{value:,.2f}"
                    
                assets_table_data.append([label, formatted_value])
        
        # Create liabilities and equity table
        liab_equity_table_data = [["Liabilities & Equity", "Value (in thousands)"]]
        
        for key, label in liabilities_equity_items:
            if key in balance_data:
                value = balance_data[key]
                
                # Format the value
                if value is None:
                    formatted_value = "N/A"
                else:
                    formatted_value = f"{value:,.2f}"
                    
                liab_equity_table_data.append([label, formatted_value])
        
        # Style the assets table
        assets_table = Table(assets_table_data, colWidths=[3*inch, 2*inch])
        assets_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.cornflowerblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
            # Highlight Total Assets
            ('BACKGROUND', (0, -1), (-1, -1), colors.lightblue),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ]))
        story.append(assets_table)
        story.append(Spacer(1, 0.15 * inch))
        
        # Style the liabilities and equity table
        liab_equity_table = Table(liab_equity_table_data, colWidths=[3*inch, 2*inch])
        liab_equity_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.cornflowerblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
            # Highlight Shareholders' Equity
            ('BACKGROUND', (0, -1), (-1, -1), colors.lightblue),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ]))
        story.append(liab_equity_table)
    
    # Add notes if available
    if "notes" in extracted_data and extracted_data["notes"]:
        notes = extracted_data["notes"]
        has_notes = False
        
        # Check if we have EBITDA or working capital notes
        if notes.get("adj_ebitda_available") or notes.get("adj_working_capital_available"):
            story.append(PageBreak())
            story.append(Paragraph("5. ADDITIONAL ANALYSIS", heading1_style))
            story.append(HorizontalRule(450, thickness=2, color=colors.lightsteelblue))
            story.append(Spacer(1, 0.15 * inch))
            
            # Add Adjusted EBITDA Analysis if available
            if notes.get("adj_ebitda_available"):
                story.append(Paragraph("5.1 Adjusted EBITDA Analysis", heading2_style))
                ebitda_details = notes.get("adj_ebitda_details", "No details provided.")
                story.append(Paragraph(ebitda_details, body_style))
                story.append(Spacer(1, 0.2 * inch))
                has_notes = True
            
            # Add Adjusted Working Capital Analysis if available
            if notes.get("adj_working_capital_available"):
                section_num = "5.2" if has_notes else "5.1"
                story.append(Paragraph(f"{section_num} Adjusted Working Capital Analysis", heading2_style))
                wc_details = notes.get("adj_working_capital_details", "No details provided.")
                story.append(Paragraph(wc_details, body_style))
                story.append(Spacer(1, 0.2 * inch))

    # --- ESG Insights Section ---
    # Check if ESG data is available before adding the page break and title
    extracted_esg_data = report_data.get("extracted_esg_data", {})
    
    # Only add ESG section if there is data or a specific error message to display for ESG
    if extracted_esg_data: 
        story.append(PageBreak())
        story.append(Paragraph("6. ESG (Environmental, Social, and Governance) Insights", heading1_style))
        story.append(HorizontalRule(450, thickness=2, color=colors.lightsteelblue))
        story.append(Spacer(1, 0.15 * inch))

        if isinstance(extracted_esg_data, dict) and "error" in extracted_esg_data:
            story.append(Paragraph(f"Could not generate ESG insights: {extracted_esg_data['error']}", body_style))
        else:
            esg_categories = {
                "environmental": "6.1 Environmental",
                "social": "6.2 Social",
                "governance": "6.3 Governance"
            }
            
            esg_data_presented = False # Flag to check if any ESG data was actually presented

            for category_key, section_title in esg_categories.items():
                sub_category_data = extracted_esg_data.get(category_key) # Use .get() for safety
                
                # Ensure sub_category_data is a dictionary before trying to iterate
                if isinstance(sub_category_data, dict) and sub_category_data:
                    story.append(Paragraph(section_title, heading2_style))
                    esg_table_data = []
                    for k, v in sub_category_data.items():
                        formatted_value = str(v) if v is not None and str(v).strip() != "" else "N/A"
                        if len(formatted_value) > 150: # Truncate long strings
                            formatted_value = formatted_value[:147] + "..."
                        esg_table_data.append([format_esg_key(k), formatted_value])
                    
                    if not esg_table_data:
                        story.append(Paragraph("No data available for this sub-category.", body_style))
                    else:
                        esg_table_data.insert(0, [format_esg_key(category_key) + " Factor", "Details"])
                        
                        esg_table = Table(esg_table_data, colWidths=[2.5*inch, 3.5*inch]) # Adjusted colWidths
                        esg_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.cornflowerblue),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, 0), 10),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                            ('ALIGN', (0, 1), (0, -1), 'LEFT'), 
                            ('ALIGN', (1, 1), (1, -1), 'LEFT'), 
                            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
                        ]))
                        story.append(esg_table)
                        esg_data_presented = True
                    story.append(Spacer(1, 0.2 * inch))
                elif sub_category_data is None or (isinstance(sub_category_data, dict) and not sub_category_data) : # Explicitly handle None or empty dict
                    story.append(Paragraph(section_title, heading2_style))
                    story.append(Paragraph("No data available for this sub-category.", body_style))
                    story.append(Spacer(1, 0.2 * inch))
            
            overall_summary = extracted_esg_data.get("overall_esg_commitments_summary")
            if overall_summary and isinstance(overall_summary, str) and overall_summary.strip():
                story.append(Paragraph("6.4 Overall ESG Commitments", heading2_style))
                story.append(Paragraph(overall_summary, body_style))
                story.append(Spacer(1, 0.2 * inch))
                esg_data_presented = True
            elif "overall_esg_commitments_summary" in extracted_esg_data and not overall_summary: # Key exists but value is None/empty
                story.append(Paragraph("6.4 Overall ESG Commitments", heading2_style))
                story.append(Paragraph("No overall ESG commitments summary provided.", body_style))
                story.append(Spacer(1, 0.2 * inch))
                esg_data_presented = True


            if not esg_data_presented and not (isinstance(extracted_esg_data, dict) and "error" in extracted_esg_data):
                # This message is if no specific ESG data points were found AND no error message was already shown.
                story.append(Paragraph("No specific ESG data points were found in the document.", body_style))
    
    # Add footnote
    story.append(Spacer(1, inch))
    footnote = "This report was automatically generated using AI-based financial data extraction and analysis. " \
               "Values are approximate and should be verified against original sources. " \
               "Professional accounting advice should be sought before making business decisions based on this report."
    story.append(Paragraph(footnote, ParagraphStyle(
        'Footnote',
        parent=styles['Italic'],
        fontSize=8,
        textColor=colors.gray
    )))
    
    # Build the document
    doc.build(story)
    
    return output_path