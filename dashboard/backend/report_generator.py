from fpdf import FPDF
from datetime import datetime
import json
import os
from pathlib import Path

class SiteReportPDF(FPDF):
    def header(self):
        # Logo placeholder (can be implemented later)
        # self.image('logo.png', 10, 8, 33)
        self.set_font('Helvetica', 'B', 16)
        self.set_text_color(44, 62, 80) # Dark blue/grey
        self.cell(0, 10, 'BUILDSIGHT SITE INTELLIGENCE REPORT', 0, 1, 'C')
        self.set_font('Helvetica', '', 10)
        self.set_text_color(127, 140, 141) # Grey
        self.cell(0, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        self.ln(5)
        # Draw a line
        self.set_draw_color(236, 240, 241)
        self.line(10, 35, 200, 35)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(149, 165, 166)
        self.cell(0, 10, f'Page {self.page_no()} | Green Build AI - BuildSight Proprietary', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 12)
        self.set_fill_color(248, 249, 250)
        self.set_text_color(44, 62, 80)
        self.cell(0, 10, f'  {title}', 0, 1, 'L', True)
        self.ln(4)

    def kpi_box(self, label, value, x, y, w=45, h=25, status='normal'):
        # Colors based on status
        colors = {
            'safe': (46, 204, 113),    # Green
            'warning': (241, 196, 15), # Amber
            'danger': (231, 76, 60),   # Red
            'normal': (52, 152, 219)   # Blue
        }
        color = colors.get(status, colors['normal'])
        
        # Border and shadow effect (subtle)
        self.set_draw_color(236, 240, 241)
        self.set_fill_color(255, 255, 255)
        self.rect(x, y, w, h, 'FD')
        
        # Color bar at top
        self.set_fill_color(*color)
        self.rect(x, y, w, 2, 'F')
        
        # Label
        self.set_xy(x + 2, y + 5)
        self.set_font('Helvetica', '', 8)
        self.set_text_color(127, 140, 141)
        self.cell(w - 4, 5, label.upper(), 0, 1, 'C')
        
        # Value
        self.set_xy(x + 2, y + 12)
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(*color)
        self.cell(w - 4, 10, str(value), 0, 1, 'C')

def generate_daily_report(site_name, date_str, data):
    """
    Generates a PDF report for a given date.
    data structure:
    {
        "summary": {"workers": 45, "compliance": 92.5, "violations": 12, "incidents": 2},
        "zones": [
            {"name": "Excavation", "workers": 12, "risk": "High", "violations": 5},
            ...
        ],
        "violation_types": {"No Helmet": 8, "No Vest": 4},
        "incidents": [
            {"time": "10:24", "zone": "Zone A", "message": "Restricted area intrusion"}
        ]
    }
    """
    pdf = SiteReportPDF()
    pdf.add_page()
    
    # --- SITE SUMMARY ---
    pdf.chapter_title(f"Daily Executive Summary - {date_str}")
    
    summaries = data.get("summary", {})
    # KPI Row 1
    pdf.kpi_box("Peak Workers", summaries.get("workers", 0), 10, 55, status='normal')
    
    comp = summaries.get("compliance", 0)
    status = 'safe' if comp > 90 else ('warning' if comp > 75 else 'danger')
    pdf.kpi_box("Compliance Score", f"{comp}%", 60, 55, status=status)
    
    viols = summaries.get("violations", 0)
    status = 'safe' if viols == 0 else ('warning' if viols < 5 else 'danger')
    pdf.kpi_box("Total Violations", viols, 110, 55, status=status)
    
    incidents = summaries.get("incidents", 0)
    status = 'safe' if incidents == 0 else 'danger'
    pdf.kpi_box("Safety Incidents", incidents, 160, 55, status=status)
    
    pdf.ln(35)
    
    # --- AI EXECUTIVE NARRATIVE ---
    ai_data = data.get("ai_narrative")
    if ai_data:
        pdf.chapter_title("Turner AI Executive Narrative")
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(44, 62, 80)
        pdf.cell(0, 6, f"Confidence Score: {ai_data.get('confidence', 0)}/100", 0, 1)
        pdf.ln(2)
        
        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(52, 73, 94)
        pdf.multi_cell(0, 6, str(ai_data.get('summary', '')))
        pdf.ln(4)
        
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(0, 6, "Supervisor Detailed Commentary:", 0, 1)
        pdf.set_font('Helvetica', '', 10)
        pdf.multi_cell(0, 6, str(ai_data.get('detailed_commentary', '')))
        pdf.ln(10)
    
    # --- RISK BY ZONE ---
    pdf.chapter_title("Safety Risk by Zone")
    
    # Table Header
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_fill_color(236, 240, 241)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(60, 10, " Zone Name", 0, 0, 'L', True)
    pdf.cell(30, 10, " Activity", 0, 0, 'C', True)
    pdf.cell(30, 10, " Violations", 0, 0, 'C', True)
    pdf.cell(30, 10, " Comp %", 0, 0, 'C', True)
    pdf.cell(40, 10, " Risk Level ", 0, 1, 'R', True)
    
    # Table Data
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(52, 73, 94)
    
    zones = data.get("zones", [])
    if not zones:
        pdf.cell(0, 10, "No zone data available for this period.", 0, 1, 'C')
    else:
        for zone in zones:
            pdf.cell(60, 10, f" {zone.get('zone_name', 'Unknown')}", 'B', 0, 'L')
            pdf.cell(30, 10, str(zone.get('activity', 0)), 'B', 0, 'C')
            pdf.cell(30, 10, str(zone.get('violations', 0)), 'B', 0, 'C')
            comp = zone.get('compliance_pct', 100)
            pdf.cell(30, 10, f"{comp}%", 'B', 0, 'C')
            
            # Risk Level Color coding
            risk = zone.get('risk_level', 'Low')
            if risk == 'High':
                pdf.set_text_color(231, 76, 60)
            elif risk == 'Medium':
                pdf.set_text_color(241, 196, 15)
            else:
                pdf.set_text_color(46, 204, 113)
            
            pdf.cell(40, 10, f"{risk} ", 'B', 1, 'R')
            pdf.set_text_color(52, 73, 94) # Reset
            
    pdf.ln(10)

    
    # --- VIOLATION BREAKDOWN ---
    pdf.chapter_title("Violation Distribution")
    
    v_stats = data.get("violation_types", {})
    if not v_stats:
        pdf.cell(0, 10, "No violations recorded.", 0, 1, 'C')
    else:
        # Simple vertical bar chart simulation or list
        max_val = max(v_stats.values()) if v_stats else 1
        for v_type, count in v_stats.items():
            pdf.set_font('Helvetica', '', 10)
            pdf.cell(50, 8, v_type, 0, 0)
            
            # Draw bar
            bar_w = (count / max_val) * 100
            pdf.set_fill_color(231, 76, 60) # Red bars for violations
            pdf.rect(60, pdf.get_y() + 1, bar_w, 6, 'F')
            
            pdf.set_x(170)
            pdf.cell(20, 8, str(count), 0, 1, 'R')
            
    pdf.ln(10)
    
    # --- INCIDENT LOG & HOURLY TRENDS ---
    pdf.chapter_title("Recent Alerts & Hourly Safety Trends")
    
    # Hourly Trends Table
    trends = data.get("hourly_trends", [])
    if trends:
        pdf.set_font('Helvetica', 'B', 9)
        pdf.set_fill_color(248, 249, 250)
        pdf.cell(40, 8, " Time", 'B', 0, 'L', True)
        pdf.cell(40, 8, " Active Workers", 'B', 0, 'C', True)
        pdf.cell(40, 8, " Violations", 'B', 0, 'C', True)
        pdf.cell(40, 8, " Period Risk", 'B', 1, 'C', True)
        
        pdf.set_font('Helvetica', '', 9)
        for t in trends:
            pdf.cell(40, 8, f"  {t.get('hour', '')}", 'B', 0, 'L')
            pdf.cell(40, 8, str(t.get('workers', 0)), 'B', 0, 'C')
            pdf.cell(40, 8, str(t.get('violations', 0)), 'B', 0, 'C')
            
            risk = t.get('risk', 'Low')
            if risk in ['High', 'Critical']:
                pdf.set_text_color(231, 76, 60)
            elif risk == 'Medium':
                pdf.set_text_color(241, 196, 15)
            else:
                pdf.set_text_color(46, 204, 113)
                
            pdf.cell(40, 8, risk, 'B', 1, 'C')
            pdf.set_text_color(52, 73, 94)
            
        pdf.ln(5)
    
    log = data.get("incidents", [])
    if not log:
        pdf.set_font('Helvetica', 'I', 10)
        pdf.cell(0, 10, "No incidents reported.", 0, 1, 'L')
    else:
        pdf.set_font('Helvetica', 'B', 9)
        pdf.cell(0, 8, "Latest Incident Logs:", 0, 1)
        pdf.set_font('Helvetica', '', 9)
        for entry in log:
            pdf.set_text_color(231, 76, 60) # Red bullet
            pdf.cell(5, 6, chr(149), 0, 0)
            pdf.set_text_color(52, 73, 94)
            pdf.cell(0, 6, f"[{entry.get('time', '--:--')}] {entry.get('zone', 'Global')}: {entry.get('message', '')}", 0, 1)
            
    pdf.ln(10)
    
    # --- WORKER INTELLIGENCE ---
    pdf.chapter_title("Worker Intelligence & Repeat Offenders")
    wi = data.get("worker_intelligence", [])
    if not wi:
        pdf.set_font('Helvetica', 'I', 10)
        pdf.cell(0, 10, "No persistent repeat offenders tracked locally.", 0, 1, 'C')
    else:
        pdf.set_font('Helvetica', 'B', 9)
        pdf.set_fill_color(236, 240, 241)
        pdf.cell(40, 8, " Worker Track ID", 0, 0, 'C', True)
        pdf.cell(30, 8, " Offenses", 0, 0, 'C', True)
        pdf.cell(60, 8, " Primary Offense Pattern", 0, 0, 'C', True)
        pdf.cell(40, 8, " Time in Unsafe Zones", 0, 1, 'C', True)
        
        pdf.set_font('Helvetica', '', 9)
        for w in wi:
            pdf.cell(40, 8, str(w.get('worker_id', '')), 'B', 0, 'C')
            pdf.set_text_color(231, 76, 60)
            pdf.cell(30, 8, str(w.get('offenses', 0)), 'B', 0, 'C')
            pdf.set_text_color(52, 73, 94)
            pdf.cell(60, 8, str(w.get('primary_offense', '')), 'B', 0, 'C')
            pdf.cell(40, 8, str(w.get('time_in_unsafe', '')), 'B', 1, 'C')

    # Save to buffer
    return pdf.output()

if __name__ == "__main__":
    # Test generation
    test_data = {
        "summary": {"workers": 45, "compliance": 92.5, "violations": 12, "incidents": 2},
        "zones": [
            {"name": "Excavation Sector", "workers": 12, "risk": "High", "violations": 5},
            {"name": "Material Storage", "workers": 8, "risk": "Low", "violations": 0},
            {"name": "Restricted Area A", "workers": 2, "risk": "High", "violations": 7},
            {"name": "General Level 1", "workers": 23, "risk": "Medium", "violations": 0}
        ],
        "violation_types": {
            "No Helmet": 8,
            "No Safety Vest": 4,
            "Zone Intrusion": 2
        },
        "incidents": [
            {"time": "09:45", "zone": "Zone A", "message": "Worker entered restricted zone without permit"},
            {"time": "14:12", "zone": "Excavation", "message": "PPE violation detected: Missing Helmet"}
        ]
    }
    
    output = generate_daily_report("BuildSight Site Alpha", "2026-04-18", test_data)
    with open("test_report.pdf", "wb") as f:
        f.write(output)
    print("Test report generated: test_report.pdf")
