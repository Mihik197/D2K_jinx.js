# report_generator.py

import logging
from reportlab.lib.pagesizes import LETTER, A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    PageBreak, Flowable, KeepTogether, ListFlowable, ListItem # Added KeepTogether, TOC, ListFlowable, ListItem
)
from reportlab.platypus.tableofcontents import TableOfContents
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
from reportlab.pdfgen import canvas # Import canvas directly
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT # For alignments

matplotlib.use('Agg') # Use non-interactive backend

# --- Canvas for Page Numbers and TOC ---
class NumberedCanvas(canvas.Canvas):
    """
    Custom canvas to handle page numbers and TOC entries.
    Requires ReportLab >= 3.6 for reliable multiBuild
    """
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []
        self.toc_entries = [] # Store TOC entries [(level, text, pageNum, key)]
        self._page_num = 1 # Initialize page number

    def showPage(self):
        # Store state before starting new page
        self._saved_page_states.append(dict(self.__dict__))
        # Reset relevant state for the new page
        self._startPage()

    def save(self):
        """Generate page numbers and finalize TOC after build."""
        num_pages = len(self._saved_page_states)
        toc_page_map = {entry[3]: entry[2] for entry in self.toc_entries} # key -> pageNum

        for i, state in enumerate(self._saved_page_states):
            self.__dict__.update(state) # Restore state for page i+1
            self.draw_page_header_footer(i + 1, num_pages, state.get('_page_info', {})) # Draw header/footer
            canvas.Canvas.showPage(self) # Render the page
        canvas.Canvas.save(self)

    def draw_page_header_footer(self, page_num, page_count, page_info):
        """Draws the header and footer for each page."""
        self.saveState()
        self.setFont('Helvetica', 9)
        self.setFillColor(colors.grey)

        # Footer - Page Number
        footer_text = f"Page {page_num} of {page_count}"
        self.drawRightString(self._pagesize[0] - 0.75 * inch, 0.75 * inch, footer_text)

        # Footer - Generation Date
        gen_date = datetime.now().strftime('%Y-%m-%d %H:%M')
        self.drawString(0.75 * inch, 0.75 * inch, f"Generated: {gen_date}")

        # Header - Report Title (Example)
        # You might want to pass the company name here via page_info if needed
        company_name = page_info.get("company_name", "Financial Analysis Report")
        self.setFont('Helvetica-Bold', 10)
        self.setFillColor(colors.darkblue)
        self.drawCentredString(self._pagesize[0] / 2, self._pagesize[1] - 0.75 * inch, company_name.upper())

        # Header/Footer Lines
        self.setStrokeColor(colors.lightgrey)
        self.setLineWidth(0.5)
        self.line(0.75 * inch, self._pagesize[1] - 0.9 * inch, self._pagesize[0] - 0.75 * inch, self._pagesize[1] - 0.9 * inch) # Below header
        self.line(0.75 * inch, 0.9 * inch, self._pagesize[0] - 0.75 * inch, 0.9 * inch) # Above footer

        self.restoreState()

    def handle_toc_entry(self, level, text, pageNum, key):
        """Callback to store TOC entries during the build."""
        # print(f"TOC Entry: Level={level}, Text='{text}', Page={pageNum}, Key='{key}'") # Debugging
        self.toc_entries.append((level, text, pageNum, key))

    # --- Standard Flowable Methods (needed for ReportLab interaction) ---
    def beginText(self, x, y):
        # Pass through to underlying canvas
        return self._canvas.beginText(x, y)

    def drawCentredString(self, x, y, text):
         self.drawCentredString(x, y, text) # Call own method for consistency if overloaded

    def drawRightString(self, x, y, text):
         self.drawRightString(x, y, text) # Call own method for consistency if overloaded

    def drawString(self, x, y, text):
         self.drawString(x, y, text) # Call own method for consistency if overloaded

    # Add other canvas methods if your flowables need them...


# --- Helper Flowables ---
class HorizontalRule(Flowable):
    """A horizontal line flowable."""
    def __init__(self, width, thickness=1, color=colors.black, v_offset=0):
        Flowable.__init__(self)
        self.width = width
        self.thickness = thickness
        self.color = color
        self.v_offset = v_offset # Vertical offset from baseline

    def draw(self):
        self.canv.saveState()
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, self.v_offset, self.width, self.v_offset)
        self.canv.restoreState()

    def wrap(self, availWidth, availHeight):
        # Takes up horizontal space but minimal vertical space
        return (self.width, self.thickness + abs(self.v_offset) * 2)


# --- Styling ---
def get_report_styles():
    """Returns a dictionary of ParagraphStyles for the report."""
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='ReportTitle', parent=styles['h1'], fontSize=22, alignment=TA_CENTER,
        textColor=colors.darkblue, spaceAfter=18, fontName='Helvetica-Bold'
    ))
    styles.add(ParagraphStyle(
        name='ReportSubTitle', parent=styles['Normal'], fontSize=14, alignment=TA_CENTER,
        textColor=colors.darkgrey, spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        name='GeneratedDate', parent=styles['Normal'], fontSize=9, alignment=TA_CENTER,
        textColor=colors.grey, spaceAfter=30
    ))
    styles.add(ParagraphStyle(
        name='Disclaimer', parent=styles['Italic'], fontSize=8, alignment=TA_CENTER,
        textColor=colors.grey, spaceBefore=20
    ))
    styles.add(ParagraphStyle(
        name='H1', parent=styles['h1'], fontSize=16, spaceBefore=18, spaceAfter=10,
        textColor=colors.darkblue, fontName='Helvetica-Bold', keepWithNext=1, # Keep H1 with next paragraph
        # TOC level 0
    ))
    styles.add(ParagraphStyle(
        name='H2', parent=styles['h2'], fontSize=14, spaceBefore=12, spaceAfter=6,
        textColor=colors.darkslategray, fontName='Helvetica-Bold', keepWithNext=1,
        # TOC level 1
        leftIndent=12 # Indent H2
    ))
    styles.add(ParagraphStyle(
        name='H3', parent=styles['h3'], fontSize=12, spaceBefore=10, spaceAfter=4,
        textColor=colors.black, fontName='Helvetica-Bold', keepWithNext=1,
        # TOC level 2
        leftIndent=24 # Indent H3
    ))
    styles.add(ParagraphStyle(
        name='BodyText', parent=styles['Normal'], fontSize=10, leading=14, spaceAfter=6,
        alignment=TA_LEFT
    ))
    styles.add(ParagraphStyle(
        name='InfoBox', parent=styles['BodyText'], backColor=colors.lightyellow,
        borderWidth=1, borderColor=colors.orange, borderPadding=8, borderRadius=5,
        spaceAfter=12
    ))
    styles.add(ParagraphStyle(
        name='RedFlagBox', parent=styles['BodyText'], backColor=colors.mistyrose,
        borderWidth=1, borderColor=colors.red, borderPadding=8, borderRadius=5,
        spaceAfter=12
    ))
    styles.add(ParagraphStyle(
        name='ListItem', parent=styles['BodyText'], leftIndent=18, bulletIndent=0, spaceAfter=3
    ))
    styles.add(ParagraphStyle(
        name='TableValue', parent=styles['Normal'], alignment=TA_RIGHT, fontSize=9
    ))
    styles.add(ParagraphStyle(
        name='TableCenterValue', parent=styles['Normal'], alignment=TA_CENTER, fontSize=9
    ))
    styles.add(ParagraphStyle(
        name='TableHeader', parent=styles['Normal'], alignment=TA_CENTER, fontSize=10,
        textColor=colors.whitesmoke, fontName='Helvetica-Bold'
    ))
    styles.add(ParagraphStyle(
        name='ChartTitle', parent=styles['Normal'], alignment=TA_CENTER, fontSize=11,
        fontName='Helvetica-Bold', spaceBefore=6, spaceAfter=4
    ))
    return styles


# --- Charting Functions (modified for consistency) ---

def format_financial_value(value, currency="₹", scale_hint="auto"):
    """Formats numerical values for display, attempting to handle scale."""
    if value is None or value == "N/A" or not isinstance(value, (int, float)):
        return "N/A"
    try:
        # Basic auto-detection (can be improved)
        if scale_hint == "auto":
            if abs(value) >= 1_000_000_000: # Billions
                 scale_hint = 'billions'
            elif abs(value) >= 1_000_000: # Millions
                 scale_hint = 'millions'
            elif abs(value) >= 1_000: # Thousands
                 scale_hint = 'thousands'
            else:
                 scale_hint = 'units'

        # Apply scaling
        if scale_hint == 'billions':
             return f"{currency}{value/1_000_000_000:.2f}B"
        elif scale_hint == 'millions':
             return f"{currency}{value/1_000_000:.2f}M"
        elif scale_hint == 'thousands':
             return f"{currency}{value/1_000:.2f}K"
        else: # Units
             return f"{currency}{value:.2f}"

    except (TypeError, ValueError):
        return str(value) # Fallback

def create_pie_chart(data, title: str, styles):
    """Creates a ReportLab Image flowable from a Matplotlib pie chart."""
    filtered_data = {k: abs(v) for k, v in data.items() if isinstance(v, (int, float)) and v != 0}
    if not filtered_data or sum(filtered_data.values()) == 0:
        return Paragraph(f"<i>No data available for '{title}' chart.</i>", styles['BodyText'])

    labels = filtered_data.keys()
    sizes = filtered_data.values()
    colors_list = plt.cm.Paired(np.linspace(0, 1, len(labels))) # Use a standard color map

    fig, ax = plt.subplots(figsize=(6, 4)) # Adjusted size
    wedges, texts, autotexts = ax.pie(sizes, autopct='%1.1f%%', startangle=90, colors=colors_list,
                                      wedgeprops={'edgecolor': 'white', 'linewidth': 0.5})
    ax.axis('equal') # Equal aspect ratio ensures a circular pie chart

    # Improve label/legend handling
    ax.legend(wedges, labels, title="Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize='small')
    plt.setp(autotexts, size=8, weight="bold", color="white") # Text on wedges

    # No title on the chart itself, use Paragraph above
    # plt.title(title, fontsize=12, fontweight='bold')

    img_data = io.BytesIO()
    plt.savefig(img_data, format='PNG', dpi=150, bbox_inches='tight')
    plt.close(fig)
    img_data.seek(0)

    return Image(img_data, width=4*inch, height=2.67*inch) # Adjust image size

def create_bar_chart(data, title: str, styles, y_label="Value", bar_colors=None):
    """Creates a ReportLab Image flowable from a Matplotlib bar chart."""
    valid_data = {k: v for k, v in data.items() if isinstance(v, (int, float))}
    if not valid_data:
        return Paragraph(f"<i>No data available for '{title}' chart.</i>", styles['BodyText'])

    labels = list(valid_data.keys())
    values = list(valid_data.values())

    if bar_colors is None:
        bar_colors = plt.cm.Viridis(np.linspace(0.4, 0.9, len(values))) # Default color map

    fig, ax = plt.subplots(figsize=(7, 4)) # Adjusted size
    bars = ax.bar(labels, values, color=bar_colors)

    ax.set_ylabel(y_label, fontsize=9)
    # ax.set_title(title, fontsize=12, fontweight='bold') # Title handled by Paragraph
    ax.tick_params(axis='x', rotation=30, labelsize=8, ha='right')
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    # Add data labels
    for bar in bars:
        yval = bar.get_height()
        va = 'bottom' if yval >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}',
                va=va, ha='center', fontsize=7)

    plt.tight_layout() # Adjust layout

    img_data = io.BytesIO()
    plt.savefig(img_data, format='PNG', dpi=150) # bbox_inches='tight' removed for potentially better layout
    plt.close(fig)
    img_data.seek(0)

    return Image(img_data, width=5.5*inch, height=3.14*inch) # Adjust image size


# --- Main PDF Generation Function (COMPLETE) ---
def generate_pdf_report(report_data: dict, output_path: str, canvas_maker=NumberedCanvas):
    """
    Generates a comprehensive PDF financial report with automatic TOC.
    """
    log = logging.getLogger(__name__) # Use specific logger
    log.info(f"Starting PDF generation for: {output_path}")

    doc = SimpleDocTemplate(output_path, pagesize=LETTER,
                            leftMargin=0.75*inch, rightMargin=0.75*inch,
                            topMargin=1.2*inch, bottomMargin=1.2*inch)
    styles = get_report_styles()
    story = []

    # --- Document Info ---
    extracted_data = report_data.get("extracted_data", {})
    company_name = extracted_data.get("company_name", "Financial Analysis")
    reporting_period = extracted_data.get("reporting_period", "N/A")
    currency_symbol = extracted_data.get("currency", "$")
    doc.page_info = {"company_name": company_name} # Pass info to canvas

    log.info(f"Report for: {company_name}, Period: {reporting_period}")

    # --- Title Page ---
    story.append(Paragraph(f"{company_name}", styles['ReportTitle']))
    story.append(Paragraph(f"Financial Analysis Report", styles['ReportSubTitle']))
    story.append(Paragraph(f"Reporting Period: {reporting_period}", styles['ReportSubTitle']))
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", styles['GeneratedDate']))
    story.append(Spacer(1, 1.5 * inch))
    story.append(HorizontalRule(doc.width, thickness=0.5, color=colors.lightgrey))
    story.append(Paragraph(
        "Disclaimer: This report was automatically generated using AI. Financial data and analysis require verification against original sources before making business decisions. Consult with qualified professionals.",
        styles['Disclaimer']
    ))
    story.append(PageBreak())

    # --- Table of Contents ---
    toc = TableOfContents()
    # Styling the TOC (Level 0: H1, Level 1: H2, Level 2: H3)
    # Make sure names match the heading styles used ('H1', 'H2', 'H3')
    toc.levelStyles = [
        ParagraphStyle(name='TOC_H1', parent=styles['Normal'], fontSize=12, leftIndent=10, spaceBefore=6, fontName='Helvetica-Bold'),
        ParagraphStyle(name='TOC_H2', parent=styles['Normal'], fontSize=10, leftIndent=25, spaceBefore=3),
        ParagraphStyle(name='TOC_H3', parent=styles['Normal'], fontSize=9, leftIndent=40, spaceBefore=1)
    ]
    # Add TOC Header and TOC placeholder
    story.append(Paragraph("Table of Contents", styles['H1'])) # Use H1 style, no bookmark needed
    story.append(HorizontalRule(doc.width, color=colors.darkblue))
    story.append(Spacer(1, 0.2*inch))
    story.append(toc)
    story.append(PageBreak())

    # --- Report Sections ---

    # 1. Executive Summary
    story.append(KeepTogether([
        Paragraph("1. Executive Summary", styles['H1'], bookmarkName='exec_summary'), # Bookmark for TOC link
        HorizontalRule(doc.width, color=colors.darkblue),
        Spacer(1, 0.1*inch),
        Paragraph("<b>Key Findings Summary:</b>", styles['H2']), # Use H2 style for subheadings within summary
        Paragraph(report_data.get("key_findings", "Analysis not available."), styles['BodyText']),
        Spacer(1, 0.1*inch),
        Paragraph("<b>Management Sentiment Summary:</b>", styles['H2']),
        Paragraph(report_data.get("sentiment_analysis", "Analysis not available."), styles['BodyText']),
         Spacer(1, 0.1*inch),
        Paragraph("<b>Business Model Suggestions Summary:</b>", styles['H2']),
        Paragraph(report_data.get("business_model", "Analysis not available."), styles['BodyText']),
    ]))
    story.append(PageBreak())

    # 2. Detailed Business Overview
    story.append(KeepTogether([
        Paragraph("2. Business Overview", styles['H1'], bookmarkName='biz_overview'),
        HorizontalRule(doc.width, color=colors.darkblue),
        Paragraph(report_data.get("business_overview", "Overview analysis not available."), styles['BodyText']),
    ]))
    story.append(Spacer(1, 0.2*inch))

    # Key Metrics Snapshot Box
    income_statement = extracted_data.get("income_statement", {})
    balance_sheet = extracted_data.get("balance_sheet", {})
    if income_statement or balance_sheet:
        story.append(Paragraph("Key Metrics Snapshot", styles['H2'])) # Use H2 style
        key_metrics_data = []
        # Safely get values and format them
        if income_statement.get("net_sales") is not None:
             key_metrics_data.append(["Revenue", format_financial_value(income_statement["net_sales"], currency=currency_symbol)])
        if income_statement.get("net_income") is not None:
             key_metrics_data.append(["Net Income", format_financial_value(income_statement["net_income"], currency=currency_symbol)])
        if balance_sheet.get("total_assets") is not None:
             key_metrics_data.append(["Total Assets", format_financial_value(balance_sheet["total_assets"], currency=currency_symbol)])
        if balance_sheet.get("shareholders_equity") is not None:
             key_metrics_data.append(["Shareholder's Equity", format_financial_value(balance_sheet["shareholders_equity"], currency=currency_symbol)])

        if key_metrics_data:
             num_metrics = len(key_metrics_data)
             table_rows = []
             for i in range(0, num_metrics, 2):
                  row = []
                  # Metric 1 - Label (Bold) and Value
                  row.append(Paragraph(f"<b>{key_metrics_data[i][0]}</b>", styles['BodyText']))
                  row.append(Paragraph(key_metrics_data[i][1], styles['TableValue']))
                  # Metric 2 (if exists) - Label (Bold) and Value
                  if i + 1 < num_metrics:
                       row.append(Paragraph(f"<b>{key_metrics_data[i+1][0]}</b>", styles['BodyText']))
                       row.append(Paragraph(key_metrics_data[i+1][1], styles['TableValue']))
                  else:
                       # Add empty placeholders if odd number of metrics
                       row.extend([Paragraph("", styles['BodyText']), Paragraph("", styles['TableValue'])])
                  table_rows.append(row)

             # Create the table with 4 columns
             metrics_table = Table(table_rows, colWidths=[1.6*inch, 1.4*inch, 1.6*inch, 1.4*inch]) # Adjusted widths
             metrics_table.setStyle(TableStyle([
                 ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
                 ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                 ('BACKGROUND', (0,0), (-1,-1), colors.whitesmoke),
                 ('LEFTPADDING', (0,0), (-1,-1), 6),
                 ('RIGHTPADDING', (0,0), (-1,-1), 6),
                 # Span background across label/value pairs if needed, or keep simple grid
             ]))
             story.append(metrics_table)
             story.append(Spacer(1, 0.2*inch))

    # 3. Detailed Key Findings
    story.append(KeepTogether([
        Paragraph("3. Detailed Financial Findings", styles['H1'], bookmarkName='findings_detail'),
        HorizontalRule(doc.width, color=colors.darkblue),
        Paragraph(report_data.get("key_findings", "Detailed findings analysis not available."), styles['BodyText']),
    ]))
    story.append(PageBreak())

    # 4. Detailed Sentiment Analysis
    story.append(KeepTogether([
        Paragraph("4. Sentiment Analysis", styles['H1'], bookmarkName='sentiment'),
        HorizontalRule(doc.width, color=colors.darkblue),
        Paragraph(report_data.get("sentiment_analysis", "Sentiment analysis not available."), styles['BodyText']),
    ]))
    story.append(Spacer(1, 0.2*inch)) # Add space after section

    # 5. Detailed Business Model Opportunities
    story.append(KeepTogether([
        Paragraph("5. Business Model Opportunities", styles['H1'], bookmarkName='biz_model'),
        HorizontalRule(doc.width, color=colors.darkblue),
        Paragraph(report_data.get("business_model", "Business model suggestions not available."), styles['BodyText']),
    ]))
    story.append(PageBreak())

    # 6. Financial Ratio Analysis
    ratios = report_data.get("calculated_ratios", {})
    ratio_categories = { # Define categories locally
        "Profitability": ["Gross Margin Ratio", "Operating Margin Ratio", "Return on Assets Ratio", "Return on Equity Ratio"],
        "Liquidity": ["Current Ratio", "Cash Ratio"],
        "Solvency": ["Debt to Equity Ratio", "Debt Ratio", "Interest Coverage Ratio"],
        "Efficiency": ["Asset Turnover Ratio", "Inventory Turnover Ratio", "Receivables Turnover Ratio"]
    }
    category_explanations = { # Define explanations locally
        "Profitability": "Measures ability to generate profit relative to revenue, assets, equity.",
        "Liquidity": "Measures ability to meet short-term debt obligations.",
        "Solvency": "Measures ability to meet long-term debt obligations and financial leverage.",
        "Efficiency": "Measures how effectively the company utilizes its assets and manages operations."
    }
    if ratios:
        story.append(Paragraph("6. Financial Ratio Analysis", styles['H1'], bookmarkName='ratios'))
        story.append(HorizontalRule(doc.width, color=colors.darkblue))
        story.append(Paragraph("Analysis of key ratios across different financial dimensions.", styles['BodyText']))
        story.append(Spacer(1, 0.1*inch))

        for i, (category, ratio_names) in enumerate(ratio_categories.items()):
            category_ratios_data = []
            category_chart_data = {}

            for name in ratio_names:
                # Check if ratio exists and has a value
                if name in ratios and ratios[name].get("ratio_value") is not None:
                    ratio_info = ratios[name]
                    value = ratio_info.get("ratio_value", "N/A")
                    # Format value for table
                    formatted_value = f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
                    category_ratios_data.append([Paragraph(name, styles['BodyText']), Paragraph(formatted_value, styles['TableValue'])])
                    # Add numeric values to chart data
                    if isinstance(value, (int, float)):
                        # Use shorter names for chart labels if desired
                        chart_label = name.replace(" Ratio", "").replace(" Turnover", " Turn.")
                        category_chart_data[chart_label] = value

            if category_ratios_data:
                section_bookmark = f"ratio_{category.lower()}"
                category_content = [
                    Paragraph(f"6.{i+1} {category} Ratios", styles['H2'], bookmarkName=section_bookmark),
                    Paragraph(category_explanations.get(category, ""), styles['BodyText']),
                    Spacer(1, 0.1*inch),
                    Table(category_ratios_data, colWidths=[3.5*inch, 1.5*inch], style=TableStyle([
                        ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
                        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                        ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.whitesmoke, colors.white]),
                        ('LEFTPADDING', (0,0), (-1,-1), 6),
                        ('RIGHTPADDING', (0,0), (-1,-1), 6),
                    ])),
                    Spacer(1, 0.1*inch)
                ]
                # Add chart if data exists
                if category_chart_data:
                     chart_title = Paragraph(f"{category} Ratios Chart", styles['ChartTitle'])
                     chart_image = create_bar_chart(category_chart_data, f"{category} Ratios", styles)
                     category_content.append(chart_title)
                     category_content.append(chart_image)

                category_content.append(Spacer(1, 0.2*inch))
                story.append(KeepTogether(category_content)) # Keep category table and chart together

        # Add Pie chart for expense breakdown if data available
        if income_statement.get("net_sales") and income_statement.get("cost_of_goods_sold") and income_statement.get("operating_expenses"):
            expense_data = {
                "Cost of Goods Sold": income_statement.get("cost_of_goods_sold", 0),
                "Operating Expenses": income_statement.get("operating_expenses", 0),
                # Calculate implied 'Other Expenses/Taxes/Profit' as remainder
                "Net Income & Other": income_statement.get("net_sales") - income_statement.get("cost_of_goods_sold", 0) - income_statement.get("operating_expenses", 0)
            }
            # Only include positive values
            valid_expense_data = {k:v for k,v in expense_data.items() if v > 0}
            if valid_expense_data:
                pie_chart_title = Paragraph("Revenue & Expense Breakdown", styles['H2']) # Use H2 style
                pie_chart = create_pie_chart(valid_expense_data, "Revenue Breakdown", styles)
                story.append(KeepTogether([pie_chart_title, pie_chart]))
                story.append(Spacer(1, 0.2 * inch))

        story.append(PageBreak()) # Page break after all ratios/charts


    # 7. Financial Statements Summary
    if extracted_data and (extracted_data.get('income_statement') or extracted_data.get('balance_sheet')):
        story.append(Paragraph("7. Financial Statements Summary", styles['H1'], bookmarkName='statements'))
        story.append(HorizontalRule(doc.width, color=colors.darkblue))
        story.append(Paragraph("Extracted key figures from the financial statements.", styles['BodyText']))
        story.append(Spacer(1, 0.1*inch))

        # Determine scale based on typical values (e.g., Net Sales)
        net_sales_val = extracted_data.get('income_statement', {}).get('net_sales')
        table_scale = "auto"
        header_unit = "(Value)"
        if isinstance(net_sales_val, (int, float)):
             if abs(net_sales_val) >= 1_000_000_000: table_scale, header_unit = "billions", "(in Billions)"
             elif abs(net_sales_val) >= 1_000_000: table_scale, header_unit = "millions", "(in Millions)"
             elif abs(net_sales_val) >= 1_000: table_scale, header_unit = "thousands", "(in Thousands)"
             else: table_scale = "units" # Keep header as "(Value)"

        # Income Statement Table
        income_statement = extracted_data.get('income_statement', {})
        if income_statement:
             income_table_data = [[Paragraph("Income Statement Item", styles['TableHeader']), Paragraph(header_unit, styles['TableHeader'])]]
             # Define order locally
             key_order_income = ["net_sales", "cost_of_goods_sold", "gross_profit", "operating_expenses", "operating_income", "interest_expenses", "income_before_tax", "income_tax_expense", "net_income", "previous_year_sales", "previous_year_net_income"]
             for key in key_order_income:
                 if key in income_statement and income_statement[key] is not None: # Check if key exists and has value
                     label = " ".join(word.capitalize() for word in key.split("_"))
                     value = format_financial_value(income_statement[key], currency=currency_symbol, scale_hint=table_scale)
                     income_table_data.append([Paragraph(label, styles['BodyText']), Paragraph(value, styles['TableValue'])])
             # Add remaining items dynamically (optional)

             income_table = Table(income_table_data, colWidths=[3.5*inch, 2.5*inch], repeatRows=1) # Adjusted width
             income_table.setStyle(TableStyle([
                 ('BACKGROUND', (0,0), (-1,0), colors.darkslategray),
                 ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                 ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                 ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.whitesmoke, colors.white]),
                 ('BOTTOMPADDING', (0,0), (-1,0), 8),
                 ('TOPPADDING', (0,0), (-1,0), 8),
                 ('LEFTPADDING', (0,0), (-1,-1), 6),
                 ('RIGHTPADDING', (0,0), (-1,-1), 6),
                 # Highlight Net Income if present
                 ('ROWBACKGROUNDS', (0, -1), (-1,-1), [colors.lightsteelblue]), # Highlight last row if it's NI
                 # ('FONTNAME', (0,-1), (-1,-1), 'Helvetica-Bold'), # Optional bold for last row
             ]))
             story.append(KeepTogether([
                Paragraph("7.1 Income Statement", styles['H2'], bookmarkName='income_stmt'),
                income_table
             ]))
             story.append(Spacer(1, 0.2*inch))

        # Balance Sheet Table
        balance_sheet = extracted_data.get('balance_sheet', {})
        if balance_sheet:
             assets_data = [[Paragraph("Assets", styles['TableHeader']), Paragraph(header_unit, styles['TableHeader'])]]
             liab_equity_data = [[Paragraph("Liabilities & Equity", styles['TableHeader']), Paragraph(header_unit, styles['TableHeader'])]]
             # Define order locally
             asset_keys = ["cash_and_equivalents", "accounts_receivable", "inventory", "current_assets", "property_plant_equipment", "total_assets", "previous_year_total_assets"]
             liab_equity_keys = ["accounts_payable", "current_liabilities", "long_term_debt", "total_liabilities", "shareholders_equity", "total_liabilities_and_equity"]

             for key in asset_keys:
                 if key in balance_sheet and balance_sheet[key] is not None:
                     label = " ".join(word.capitalize() for word in key.split("_"))
                     value = format_financial_value(balance_sheet[key], currency=currency_symbol, scale_hint=table_scale)
                     assets_data.append([Paragraph(label, styles['BodyText']), Paragraph(value, styles['TableValue'])])

             for key in liab_equity_keys:
                 if key in balance_sheet and balance_sheet[key] is not None:
                     label = " ".join(word.capitalize() for word in key.split("_"))
                     value = format_financial_value(balance_sheet[key], currency=currency_symbol, scale_hint=table_scale)
                     liab_equity_data.append([Paragraph(label, styles['BodyText']), Paragraph(value, styles['TableValue'])])

             # Balance Sheet Equation Check
             total_assets_val = balance_sheet.get('total_assets')
             total_liab_equity_val = balance_sheet.get('total_liabilities_and_equity')
             balance_check = ""
             if isinstance(total_assets_val, (int, float)) and isinstance(total_liab_equity_val, (int, float)):
                 diff = abs(total_assets_val - total_liab_equity_val)
                 # Use a small tolerance for floating point comparisons
                 tolerance = 0.01 if table_scale == 'units' else 1 # Adjust tolerance based on scale
                 if diff < tolerance:
                     balance_check = "<para color='green' alignment='center'><i>Assets = Liabilities + Equity (Balanced)</i></para>"
                 else:
                     # Format difference using the detected scale
                     diff_str = format_financial_value(diff, currency=currency_symbol, scale_hint=table_scale)
                     balance_check = f"<para color='red' alignment='center'><i>Balance Sheet Discrepancy! Difference: {diff_str}</i></para>"

             table_style = TableStyle([ # Base style for both tables
                 ('BACKGROUND', (0,0), (-1,0), colors.darkslategray),
                 ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                 ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                 ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.whitesmoke, colors.white]),
                 ('BOTTOMPADDING', (0,0), (-1,0), 8),
                 ('TOPPADDING', (0,0), (-1,0), 8),
                 ('LEFTPADDING', (0,0), (-1,-1), 6),
                 ('RIGHTPADDING', (0,0), (-1,-1), 6),
                 # Highlight Totals
                 ('ROWBACKGROUNDS', (0, -1), (-1,-1), [colors.lightsteelblue]), # Highlight last row
                 # ('FONTNAME', (0,-1), (-1,-1), 'Helvetica-Bold'), # Optional bold
             ])
             assets_table = Table(assets_data, colWidths=[3.5*inch, 2.5*inch], repeatRows=1, style=table_style)
             liab_equity_table = Table(liab_equity_data, colWidths=[3.5*inch, 2.5*inch], repeatRows=1, style=table_style)

             story.append(KeepTogether([
                Paragraph("7.2 Balance Sheet", styles['H2'], bookmarkName='balance_sheet'),
                assets_table,
                Spacer(1, 0.1*inch),
                liab_equity_table,
                (Paragraph(balance_check, styles['BodyText']) if balance_check else Spacer(0,0.1*inch)), # Add balance check below tables
             ]))
        story.append(PageBreak())

    # 8. Red Flags / Concerns
    red_flags = report_data.get("red_flags", []) # Get combined flags
    story.append(Paragraph("8. Potential Red Flags & Concerns", styles['H1'], bookmarkName='red_flags'))
    story.append(HorizontalRule(doc.width, color=colors.darkblue))
    if red_flags:
        story.append(Paragraph("The following points indicate areas automatically flagged based on data analysis and standard checks, which may warrant further investigation:", styles['BodyText']))
        flags_list = []
        valid_flags = [flag for flag in red_flags if isinstance(flag, dict)]
        if not valid_flags:
             story.append(Paragraph("No specific red flags identified in the structured data.", styles['InfoBox']))
        else:
            log.info(f"Formatting {len(valid_flags)} red flags for PDF.")
            for i, flag in enumerate(valid_flags):
                # Use HTML-like tags for formatting within Paragraph
                flag_text = f"<b>{i+1}. Issue:</b> {flag.get('issue', 'N/A')}<br/>" \
                            f"   <b>Category:</b> {flag.get('category', 'N/A')}<br/>" \
                            f"   <b>Severity:</b> {flag.get('severity', 'N/A')}<br/>" \
                            f"   <b>Details:</b> {flag.get('details', 'N/A')}<br/>" \
                            f"   <b>Recommendation:</b> {flag.get('recommendation', 'N/A')}"
                flags_list.append(Paragraph(flag_text, styles['RedFlagBox'])) # Use RedFlagBox style
                flags_list.append(Spacer(1, 0.1*inch)) # Space between flags
            # Try to keep flags together, but allow breaks if needed
            story.extend(flags_list) # Add flags to story directly
    else:
        log.info("No red flags provided in report_data.")
        story.append(Paragraph("No significant red flags were automatically identified based on the provided data.", styles['InfoBox']))


    # --- Build the PDF ---
    log.info("Starting PDF build process...")
    def first_pass(canvas, doc):
        canvas._page_info = doc.page_info
        log.debug("PDF Build: First Pass")

    def later_passes(canvas, doc):
        canvas._page_info = doc.page_info
        canvas.handle_toc_entry = toc.handle_toc_entry
        log.debug(f"PDF Build: Later Pass (Page {canvas.getPageNumber()})")

    try:
        doc.multiBuild(story, canvasmaker=canvas_maker,
                       onFirstPage=first_pass, onLaterPages=later_passes)
        log.info(f"PDF report successfully generated at {output_path}")
    except Exception as e:
         log.exception(f"FATAL: Error building PDF document: {e}")
         # Consider writing a simple text file or basic PDF with the error
         # For now, re-raise to be handled by the backend caller
         raise

    return output_path