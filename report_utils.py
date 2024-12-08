"""
# PDF Report Generator for SUDCare QA
# Author: David Warren
# Created: Dec 7, 2024
#
# Handles report generation forQA analysis results.
# Uses matplotlib for charts and FPDF for the PDF output.
"""

from typing import Dict, List
import pandas as pd
from datetime import datetime
from fpdf import FPDF
import matplotlib.pyplot as plt
import os
import logging
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Basic logging setup
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Tweaked matplotlib to make charts look better
plt.style.use('bmh')  # Using a built-in style that's always available
plt.rcParams.update({
    'figure.figsize': [10, 6],
    'figure.dpi': 100,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.grid': True,
    'grid.alpha': 0.3
})

# Making seaborn play nice with current style
sns.set_theme(style="whitegrid", palette="deep")

class QAReport(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(left=15, top=15, right=15)
        self.add_page()
        self._create_header()
    
    def _create_header(self):
        """Puts a standard header on each page"""
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'SUDCare QA Analysis Report', ln=True, align='C')
        self.set_font('Arial', '', 11)
        self.cell(0, 10, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', ln=True, align='C')
        self.ln(5)
    
    def chapter_title(self, title: str):
        """Adds a larger heading for new chapters"""
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, title, ln=True)
        self.set_font('Arial', '', 11)
        self.ln(5)
    
    def section_title(self, title: str):
        """Adds a smaller heading for sections"""
        self.set_font('Arial', 'B', 12)
        self.cell(0, 8, title, ln=True)
        self.set_font('Arial', '', 11)
        self.ln(3)
    
    def body_text(self, text: str):
        """Handles regular text with smart line wrapping"""
        self.set_font('Arial', '', 11)
        # Calculate maximum line width
        max_width = self.w - 2 * self.l_margin
        # Split text into lines that fit within the page width
        lines = text.split('\n')
        for line in lines:
            words = line.split()
            current_line = ''
            for word in words:
                # Test if adding this word would exceed the line width
                test_line = current_line + (' ' if current_line else '') + word
                if self.get_string_width(test_line) > max_width:
                    # Current line is full, print it and start a new one
                    if current_line:
                        self.cell(0, 5, current_line.strip(), ln=True)
                        current_line = word
                    else:
                        # Word is too long for the line, need to split it
                        self.cell(0, 5, word, ln=True)
                        current_line = ''
                else:
                    current_line = test_line
            # Print the last line if there is one
            if current_line:
                self.cell(0, 5, current_line.strip(), ln=True)
        self.ln(2)
    
    def bullet_points(self, points: List[str]):
        """Makes bullet point lists"""
        self.set_font('Arial', '', 11)
        for point in points:
            self.cell(5, 5, "•", ln=0)
            self.cell(0, 5, point, ln=True)
    
    def add_chart(self, image_path: str, caption: str = None):
        """Drops in a chart and centers it"""
        if os.path.exists(image_path):
            # Calculate image dimensions to fit page width
            img_width = self.w - 2 * self.l_margin
            self.image(image_path, x=self.l_margin, w=img_width)
            if caption:
                self.set_font('Arial', 'I', 10)
                self.cell(0, 5, caption, ln=True, align="C")
            self.ln(5)

def save_chart(fig, filename: str) -> bool:
    """Saves matplotlib charts as PNG files"""
    try:
        logger.debug(f"Attempting to save {filename}")
        fig.savefig(filename, bbox_inches='tight', dpi=100)
        plt.close(fig)  # Close the figure to free memory
        logger.debug(f"Successfully saved {filename}")
        return True
    except Exception as e:
        logger.error(f"Error saving {filename}: {str(e)}")
        plt.close(fig)  # Make sure to close even on error
        return False


def generate_charts(df: pd.DataFrame, analysis_results: Dict) -> List[str]:
    """Creates all the charts we need for our report"""
    chart_files = []
    
    try:
        logger.info("Starting chart generation...")
        
        # 1. Issues by Standard Bar Chart
        logger.debug("Generating Issues by Standard chart...")
        try:
            standard_counts = df["Linked Standard"].value_counts()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            standard_counts.plot(kind='bar', ax=ax)
            ax.set_title("Distribution of Issues Across Standards")
            ax.set_xlabel("Standard")
            ax.set_ylabel("Number of Issues")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            standards_chart = "temp_standards_chart.png"
            if save_chart(fig, standards_chart):
                chart_files.append(standards_chart)
                
        except Exception as e:
            logger.error(f"Error generating standards chart: {str(e)}")
        
        # 2. Status Distribution Pie Chart
        logger.debug("Generating Status Distribution chart...")
        try:
            status_counts = df["Status"].fillna("Open").value_counts()
            
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%')
            ax.set_title("Issue Status Distribution")
            plt.tight_layout()
            
            status_chart = "temp_status_chart.png"
            if save_chart(fig, status_chart):
                chart_files.append(status_chart)
                
        except Exception as e:
            logger.error(f"Error generating status chart: {str(e)}")
        
        # 3. Priority Areas Bar Chart
        logger.debug("Generating Priority Areas chart...")
        try:
            priority_areas = analysis_results.get("priority_areas", [])
            if priority_areas:
                areas = [area["area"] for area in priority_areas]
                scores = [area["priority_score"] for area in priority_areas]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(range(len(areas)), scores)
                ax.set_xticks(range(len(areas)))
                ax.set_xticklabels(areas, rotation=45, ha='right')
                ax.set_title("Priority Areas Analysis")
                ax.set_ylabel("Priority Score")
                plt.tight_layout()
                
                priority_chart = "temp_priority_chart.png"
                if save_chart(fig, priority_chart):
                    chart_files.append(priority_chart)
                    
        except Exception as e:
            logger.error(f"Error generating priority areas chart: {str(e)}")
        
        logger.info(f"Chart generation complete. Generated {len(chart_files)} charts.")
        return chart_files
        
    except Exception as e:
        logger.error(f"Error in chart generation: {str(e)}")
        return chart_files

def generate_report(df: pd.DataFrame, analysis_results: Dict, output_path: str = "qa_analysis_report.pdf") -> str:
    """Puts together the full PDF report with all our analysis"""
    logger.info("Starting report generation...")
    chart_files = []
    
    try:
        # Get active issues
        active_issues = df[
            (df["Merged With Issue ID"].isna()) |  # Unmerged issues
            (df["Status"] == "Merged")  # Merged groups
        ]
        
        # Generate charts first
        logger.info(f"Generating charts for {len(active_issues)} active issues...")
        chart_files = generate_charts(active_issues, analysis_results)
        
        logger.info("Creating PDF document...")
        # Create PDF
        pdf = QAReport()
        
        # Executive Summary
        logger.debug("Adding Executive Summary...")
        pdf.chapter_title("Executive Summary")
        
        # Add analysis scope explanation
        dataset_coverage = analysis_results.get("summary", {}).get("dataset_coverage", {})
        scope_text = f"""Analysis Scope:

This analysis covers {dataset_coverage.get('total_active_issues', len(active_issues))} active issues:
- {dataset_coverage.get('merged_groups', len(active_issues[active_issues['Status'] == 'Merged']))} merged issue groups
- {dataset_coverage.get('unmerged_issues', len(active_issues[active_issues['Status'] != 'Merged']))} unmerged individual issues
- {dataset_coverage.get('standards_count', len(active_issues['Linked Standard'].unique()))} quality standards evaluated

Note: To avoid redundancy, this analysis excludes individual issues that were previously merged into groups. 
Each merged group represents multiple related issues that share common patterns or root causes."""

        pdf.body_text(scope_text)
        pdf.ln(5)
        
        # Add overall assessment
        summary = analysis_results.get("summary", {})
        pdf.body_text(summary.get("overall_assessment", "No overall assessment available."))
        pdf.ln(5)
        
        # Add critical findings
        if "critical_findings" in summary:
            pdf.section_title("Critical Findings")
            findings_text = ""
            for i, finding in enumerate(summary["critical_findings"], 1):
                findings_text += f"{i}. {finding}\n"
            pdf.body_text(findings_text)
            pdf.ln(5)
        
        # Add visualizations
        if chart_files:
            pdf.chapter_title("Data Visualizations")
            for chart_file in chart_files:
                if os.path.exists(chart_file):
                    pdf.add_chart(chart_file)
        
        # Standards Analysis
        if "standards_analysis" in analysis_results:
            pdf.chapter_title("Standards Analysis")
            for standard in analysis_results["standards_analysis"]:
                pdf.section_title(f"Standard: {standard['standard']}")
                pdf.body_text(f"Priority Level: {standard['priority_level'].upper()}")
                pdf.body_text(f"Total Issues: {standard['total_issues']}")
                
                if standard.get("key_patterns"):
                    pdf.body_text("Key Patterns:")
                    patterns_text = ""
                    for i, pattern in enumerate(standard["key_patterns"], 1):
                        patterns_text += f"{i}. {pattern}\n"
                    pdf.body_text(patterns_text)
                
                if standard.get("recommendations"):
                    pdf.body_text("Recommendations:")
                    recommendations_text = ""
                    for i, rec in enumerate(standard["recommendations"], 1):
                        recommendations_text += f"{i}. {rec}\n"
                    pdf.body_text(recommendations_text)
                pdf.ln(5)
        
        # Priority Areas
        if "priority_areas" in analysis_results:
            pdf.chapter_title("Priority Areas")
            for area in analysis_results["priority_areas"]:
                pdf.section_title(area["area"])
                pdf.body_text(f"Priority Score: {area['priority_score']}/100")
                pdf.body_text(f"Impact: {area['impact']}")
                
                if area.get("affected_standards"):
                    pdf.body_text("Affected Standards:")
                    standards_text = ""
                    for i, std in enumerate(area["affected_standards"], 1):
                        standards_text += f"{i}. {std}\n"
                    pdf.body_text(standards_text)
                
                if area.get("suggested_fixes"):
                    pdf.body_text("Suggested Fixes:")
                    fixes_text = ""
                    for i, fix in enumerate(area["suggested_fixes"], 1):
                        fixes_text += f"{i}. {fix}\n"
                    pdf.body_text(fixes_text)
                pdf.ln(5)
        
        # Improvement Roadmap
        if "improvement_roadmap" in analysis_results:
            pdf.chapter_title("Improvement Roadmap")
            for phase in analysis_results["improvement_roadmap"]:
                pdf.section_title(f"Phase {phase['phase']}: {phase['focus_area']}")
                pdf.body_text(f"Complexity: {phase['complexity'].upper()}")
                pdf.body_text(f"Expected Impact: {phase['expected_impact']}")
                
                if phase.get("actions"):
                    pdf.body_text("Actions:")
                    actions_text = ""
                    for i, action in enumerate(phase["actions"], 1):
                        actions_text += f"{i}. {action}\n"
                    pdf.body_text(actions_text)
                pdf.ln(5)
        
        # Save the report
        logger.info(f"Saving report to {output_path}")
        pdf.output(output_path)
        
        # Clean up chart files
        for chart_file in chart_files:
            try:
                if os.path.exists(chart_file):
                    os.remove(chart_file)
            except Exception as e:
                logger.warning(f"Failed to remove chart file {chart_file}: {str(e)}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        # Clean up any chart files
        for chart_file in chart_files:
            try:
                if os.path.exists(chart_file):
                    os.remove(chart_file)
            except:
                pass
        raise

# What other files can import
__all__ = ['generate_report', 'generate_charts', 'save_chart', 'QAReport']
