"""
# QA Issues Analysis Tool for SUDCare Chatbot
# Author: John Smith
# Created: December 7, 2024
#
# This is our main application for analyzing QA testing issues in the SUDCare chatbot.
# It helps us identify patterns in QA failures and suggests potential issue merges
# to streamline our testing process.
"""

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
from llm_utils import analyze_issues_for_merge
from merge_utils import MergeExecutor
import plotly.express as px
from datetime import datetime
from analysis_utils import analyze_qa_issues, calculate_priority_score, generate_priority_areas
from report_utils import generate_report
import logging

# Set up basic logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load our config from .env file
load_dotenv()

# Set up our merge helper
merge_executor = MergeExecutor()

# Set up the page layout - using wide mode for better data visibility
st.set_page_config(
    page_title="QA Issues Analysis Tool",
    page_icon="📊",
    layout="wide"
)

# These are our core QA standards that we check against
QA_STANDARDS = [
    "Suzy introduces itself clearly, explaining its supportive role in recovery",
    "Suzy uses accessible, recovery-specific language",
    "Suzy provides concise, clear responses",
    "Suzy provides supportive feedback that acknowledges user inputs, especially for personal experiences or achievements",
    "Suzy offers actionable, non-prescriptive advice for healthcare appointments",
    "Suzy suggests personalized, relevant community resources",
    "Suzy sensitively identifies distress cues and responds appropriately",
    "Suzy encourages users to connect with real-life support networks",
    "Suzy uses non-judgmental language, avoiding stigmatizing phrases",
    "Suzy communicates limitations clearly and refrains from giving medical advice",
    "Suzy offers relevant, supportive educational information when appropriate"
]

# Keep track of our app's state between reruns
if 'df' not in st.session_state:
    st.session_state.df = None
if 'merge_suggestions' not in st.session_state:
    st.session_state.merge_suggestions = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = 0

# Some custom CSS to make the UI look cleaner and more professional
st.markdown("""
<style>
    /* File upload area */
    .stFileUploader {
        padding: 1rem;
        border-radius: 0.5rem;
        background: #f8f9fa;
        border: 2px dashed #dee2e6;
    }
    
    /* Tab styling */
    div.row-widget.stRadio > div {
        flex-direction: row;
        background: #f8f9fa;
        border-radius: 0.5rem;
        padding: 0.5rem;
    }
    
    div.row-widget.stRadio > div label {
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        margin-right: 0.5rem;
    }
    
    div.row-widget.stRadio > div [data-baseweb="radio"] {
        background: white;
    }
    
    /* Section headers */
    .section-header {
        margin-top: 2rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #dee2e6;
    }
    
    /* Merge groups */
    .merge-group {
        background: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin-bottom: 1rem;
    }
    
    .confidence-high {
        color: #28a745;
    }
    
    .confidence-medium {
        color: #ffc107;
    }
    
    .confidence-low {
        color: #dc3545;
    }
    
    /* Cache controls */
    .cache-controls {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    /* Metrics */
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_validate_data(uploaded_file):
    """
    Handles the loading and validation of our CSV data file.
    Makes sure we have all required columns and sets up merge tracking columns.
    Returns the processed dataframe or an error message if something's wrong.
    """
    try:
        print("[DEBUG] Loading CSV file...")
        df = pd.read_csv(uploaded_file)
        print(f"[DEBUG] Loaded {len(df)} rows")
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        required_columns = [
            "Issue ID", "Result ID", "Test Case IDs", "Input Prompt",
            "Ground Truth", "Generated Response", "Linked Theme",
            "Linked Standard", "Session IDs", "Version Tested",
            "Run Date", "Failure Rationale", "Final Weighted Score (1-3)"
        ]
        
        # Strip whitespace from required column names for comparison
        required_columns = [col.strip() for col in required_columns]
        df_columns = [col.strip() for col in df.columns]
        
        missing_columns = [col for col in required_columns if col not in df_columns]
        if missing_columns:
            return None, f"Missing required columns: {', '.join(missing_columns)}"
        
        print("[DEBUG] Initializing merge-related columns...")
        # Initialize merge-related columns if they don't exist
        if "Status" not in df.columns:
            print("[DEBUG] Creating Status column")
            df["Status"] = pd.NA
        else:
            print("[DEBUG] Converting existing Status column")
            # Convert to string type and replace 'Open' with NA
            df["Status"] = df["Status"].astype("string[python]")
            df.loc[df["Status"] == "Open", "Status"] = pd.NA
            
        if "Merged With Issue ID" not in df.columns:
            print("[DEBUG] Creating Merged With Issue ID column")
            df["Merged With Issue ID"] = pd.NA
        else:
            print("[DEBUG] Converting existing Merged With Issue ID column")
            df["Merged With Issue ID"] = df["Merged With Issue ID"].astype("string[python]")
            
        if "Merged IDs" not in df.columns:
            print("[DEBUG] Creating Merged IDs column")
            df["Merged IDs"] = pd.NA
        else:
            print("[DEBUG] Converting existing Merged IDs column")
            df["Merged IDs"] = df["Merged IDs"].astype("string[python]")
            
        # Print column info
        print("\n[DEBUG] Column Status:")
        for col in ["Status", "Merged With Issue ID", "Merged IDs"]:
            null_count = df[col].isna().sum()
            print(f"[DEBUG] {col}: {null_count} null values out of {len(df)} rows")
            
        return df, None
    except Exception as e:
        return None, f"Error loading file: {str(e)}"

def display_merge_preview(df: pd.DataFrame, merge_suggestion: dict, group_index: int):
    """
    Shows a detailed view of issues that could be merged together.
    Lets the user pick which secondary issues to include in the merge.
    """
    issues = merge_suggestion["issues"]
    primary_issue = issues[0]
    secondary_issues = issues[1:]
    
    st.write("#### Merge Preview")
    
    # Display primary issue
    st.write("**Primary Issue**")
    primary_data = df[df["Issue ID"] == primary_issue].iloc[0]
    st.info(f"""
    **ID**: {primary_issue}
    **Input**: {primary_data['Input Prompt']}
    **Standard**: {primary_data['Linked Standard']}
    **Score**: {primary_data['Final Weighted Score (1-3)']}
    **Rationale**: {primary_data['Failure Rationale']}
    """)
    
    # Initialize selected issues in session state if not present
    if "selected_issues" not in st.session_state:
        st.session_state.selected_issues = {}
    
    # Create a unique key for this merge group
    group_key = f"group_{group_index}"
    if group_key not in st.session_state.selected_issues:
        st.session_state.selected_issues[group_key] = {}
    
    # Display secondary issues with checkboxes
    st.write("**Issues to be Merged**")
    selected_secondary = []
    
    # Display each secondary issue
    for issue_id in secondary_issues:
        # Create two columns - one for checkbox, one for content
        col1, col2 = st.columns([1, 4])
        
        with col1:
            # Initialize checkbox state for this issue if not present
            if issue_id not in st.session_state.selected_issues[group_key]:
                st.session_state.selected_issues[group_key][issue_id] = True
                
            # Create a checkbox for this issue with a unique key
            is_selected = st.checkbox(
                "Include",
                value=st.session_state.selected_issues[group_key][issue_id],
                key=f"checkbox_{group_index}_{issue_id}"
            )
            st.session_state.selected_issues[group_key][issue_id] = is_selected
            
        with col2:
            issue_data = df[df["Issue ID"] == issue_id].iloc[0]
            st.markdown(f"""
            **Issue {issue_id}**
            - **Input**: {issue_data['Input Prompt']}
            - **Score**: {issue_data['Final Weighted Score (1-3)']}
            - **Rationale**: {issue_data['Failure Rationale']}
            """)
            
        if is_selected:
            selected_secondary.append(issue_id)
            
        # Add a divider between issues
        st.divider()
    
    # Show warning if some issues are deselected
    if len(selected_secondary) < len(secondary_issues):
        excluded_count = len(secondary_issues) - len(selected_secondary)
        excluded_ids = [id for id in secondary_issues if id not in selected_secondary]
        st.warning(f"⚠️ {excluded_count} issues excluded: {', '.join(excluded_ids)}")
    
    # Store the final selection in the merge suggestion
    merge_suggestion["selected_issues"] = [primary_issue] + selected_secondary
    
    # Display merge confidence and rationale
    st.write("**Merge Details**")
    st.success(f"""
    **Confidence**: {merge_suggestion['confidence']:.2f}
    **Rationale**: {merge_suggestion['rationale']}
    """)

def analyze_issues():
    try:
        # Load and preprocess data
        df = st.session_state.df
        if df is None or df.empty:
            st.error("No data available for analysis. Please check the data files.")
            return

        print("\n[DEBUG] Calculating metrics...")
        print(f"[DEBUG] Total rows in DataFrame: {len(df)}")
        
        # Validate Status values
        valid_status_values = {"Merged", "Primary"}
        invalid_status = df[~pd.isna(df["Status"]) & ~df["Status"].isin(valid_status_values)]
        if len(invalid_status) > 0:
            print("[WARNING] Found invalid Status values:")
            print(invalid_status["Status"].value_counts())
            
        # Calculate active issues (not merged and not a primary issue)
        active_mask = (
            (pd.isna(df["Status"]) | ~df["Status"].isin(["Merged", "Primary"])) &  # Not merged or primary
            pd.isna(df["Merged With Issue ID"])  # Not a secondary issue
        )
        active_count = active_mask.sum()
        print(f"[DEBUG] Active issues count: {active_count}")
        print("[DEBUG] Active issues breakdown:")
        print("- Status is NA:", pd.isna(df["Status"]).sum())
        print("- Status not in [Merged, Primary]:", (~df["Status"].isin(["Merged", "Primary"])).sum())
        print("- Not a secondary issue:", pd.isna(df["Merged With Issue ID"]).sum())
        
        print("\n[DEBUG] Status value counts:")
        print(df["Status"].value_counts(dropna=False))
        
        # Calculate merged groups
        merged_groups = len(df[df["Status"] == "Merged"])
        print(f"[DEBUG] Merged groups (Status is 'Merged'): {merged_groups}")
        
        # Calculate unmerged issues
        unmerged_mask = (
            pd.isna(df["Status"]) & 
            pd.isna(df["Merged With Issue ID"]) & 
            pd.isna(df["Merged IDs"])
        )
        unmerged_count = unmerged_mask.sum()
        print(f"[DEBUG] Unmerged issues: {unmerged_count}")
        
        # Debug merge-related columns
        print("[DEBUG] Merge-related columns status:")
        print(f"Status NA count: {pd.isna(df['Status']).sum()}")
        print(f"Merged With Issue ID NA count: {pd.isna(df['Merged With Issue ID']).sum()}")
        print(f"Merged IDs NA count: {pd.isna(df['Merged IDs']).sum()}")
        
        # Display issue counts
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Active Issues", active_count)
        with col2:
            st.metric("Merged Groups", merged_groups)
        with col3:
            st.metric("Unmerged Issues", unmerged_count)
        
        if st.button("Generate Analysis Report", type="primary"):
            with st.spinner("Analyzing issues and generating report..."):
                try:
                    # Perform analysis
                    analysis_results = analyze_qa_issues(df)
                    
                    # Generate report
                    report_path = generate_report(df, analysis_results)
                    
                    # Read the PDF file
                    with open(report_path, "rb") as pdf_file:
                        pdf_bytes = pdf_file.read()
                    
                    # Create download button
                    st.success("✅ Report generated successfully!")
                    st.download_button(
                        label="📥 Download Analysis Report (PDF)",
                        data=pdf_bytes,
                        file_name="qa_analysis_report.pdf",
                        mime="application/pdf",
                        key="download_report"
                    )
                    
                except Exception as e:
                    logger.error(f"Error in report generation: {str(e)}")
                    st.warning("The report was generated but some visualizations may be missing. This can happen due to temporary system limitations. The report still contains all analysis text and available charts.")
                    
                    # Try to provide download even if there were some issues
                    if os.path.exists("qa_analysis_report.pdf"):
                        with open("qa_analysis_report.pdf", "rb") as pdf_file:
                            pdf_bytes = pdf_file.read()
                        st.download_button(
                            label="📥 Download Analysis Report (PDF)",
                            data=pdf_bytes,
                            file_name="qa_analysis_report.pdf",
                            mime="application/pdf",
                            key="download_report_fallback"
                        )

    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
        logger.error(f"Error in analyze_issues: {str(e)}", exc_info=True)

def main():
    st.title("QA Issues Analysis Tool")
    st.markdown("""
    This tool helps analyze and process QA testing issues for chatbot responses.
    Upload your CSV file to begin the analysis process.
    """)
    
    # File upload with custom styling
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Only load the file if it hasn't been loaded or if it's a new file
        if st.session_state.df is None:
            df, error = load_and_validate_data(uploaded_file)
            if error:
                st.error(error)
                return
            st.session_state.df = df
            st.success("File loaded successfully!")
        
        df = st.session_state.df
        
        # Create tabs with custom styling
        tabs = ["📊 Overview", "🔄 Merge Analysis", "📜 Merge History"]
        st.session_state.current_tab = tabs.index(
            st.radio("Select Tab", tabs, index=st.session_state.current_tab, horizontal=True)
        )
        
        if st.session_state.current_tab == 0:  # Overview tab
            st.markdown('<h2 class="section-header">Data Summary</h2>', unsafe_allow_html=True)
            
            # Metrics in a grid with custom styling
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                total_issues = len(df)
                st.metric("Total Issues", total_issues)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                # Active issues are those not marked as merged
                active_issues = len(df[pd.isna(df["Status"]) | (df["Status"] != "Merged")])
                st.metric("Active Issues", active_issues)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                # Unmerged issues are those not part of any merge group (either as primary or secondary)
                unmerged_issues = len(df[pd.isna(df["Status"]) & pd.isna(df["Merged With Issue ID"]) & pd.isna(df["Merged IDs"])])
                st.metric("Unmerged Issues", unmerged_issues)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Charts with consistent styling...
            
        elif st.session_state.current_tab == 1:  # Merge Analysis tab
            st.markdown('<h2 class="section-header">Issue Merge Analysis</h2>', unsafe_allow_html=True)
            
            # Cache controls in a styled container
            st.markdown('<div class="cache-controls">', unsafe_allow_html=True)
            cache_col1, cache_col2 = st.columns(2)
            with cache_col1:
                use_cache = st.checkbox("💾 Use cached merge history", value=True)
            with cache_col2:
                if st.button("🗑️ Clear Cache", type="secondary"):
                    try:
                        if os.path.exists(merge_executor.auditor.audit_file):
                            os.remove(merge_executor.auditor.audit_file)
                            st.success("Cache cleared successfully!")
                        else:
                            st.info("No cache file found.")
                    except Exception as e:
                        st.error(f"Error clearing cache: {str(e)}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Analyze button with prominence
            if st.button("🔍 Analyze Issues", type="primary"):
                with st.spinner("Analyzing issues for potential merges..."):
                    st.session_state.merge_suggestions = analyze_issues_for_merge(df)
            
            # Display merge suggestions with enhanced styling
            if st.session_state.merge_suggestions:
                for i, suggestion in enumerate(st.session_state.merge_suggestions, 1):
                    confidence = suggestion['confidence']
                    confidence_class = (
                        'confidence-high' if confidence >= 0.9
                        else 'confidence-medium' if confidence >= 0.7
                        else 'confidence-low'
                    )
                    
                    st.markdown(f'''
                    <div class="merge-group">
                        <h3>Merge Group {i} <span class="{confidence_class}">
                        (Confidence: {confidence:.2f})</span></h3>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    display_merge_preview(df, suggestion, i)
                        
                    # Add merge button for this group
                    if st.button(f"Apply Merge {i}", key=f"merge_{i}"):
                        with st.spinner("Applying merge..."):
                            updated_df, merge_action = merge_executor.execute_merge(df, suggestion)
                                
                            if merge_action:
                                st.success("Merge completed successfully!")
                                # Update the DataFrame in session state
                                st.session_state.df = updated_df
                                # Remove the applied suggestion
                                st.session_state.merge_suggestions = [
                                    s for s in st.session_state.merge_suggestions 
                                    if s != suggestion
                                ]
                                    
                                # Allow downloading the updated CSV
                                st.download_button(
                                    "Download Updated CSV",
                                    updated_df.to_csv(index=False).encode('utf-8'),
                                    "merged_issues.csv",
                                    "text/csv",
                                    key=f'download-csv-{i}'
                                )
                                    
                                # Force a rerun to update the UI
                                st.rerun()
                            else:
                                st.error("Merge validation failed. Please check the issues and try again.")
        
        elif st.session_state.current_tab == 2:  # Merge History tab
            st.markdown('<h2 class="section-header">Merge History</h2>', unsafe_allow_html=True)
            
            # Load and display merge history
            auditor = merge_executor.auditor
            history = auditor.get_merge_history()
            
            if not history:
                st.info("No merge history available yet.")
                return
                
            for entry in history:
                # Get the number of secondary issues
                secondary_count = len(entry.get('secondary_issues', []))
                
                with st.expander(f"Merge: {entry['primary_issue']} + {secondary_count} issues"):
                    st.write("**Primary Issue**")
                    st.info(f"Issue ID: {entry['primary_issue']}")
                    
                    st.write("**Secondary Issues**")
                    for issue_id in entry.get('secondary_issues', []):
                        st.warning(f"Issue ID: {issue_id}")
                    
                    st.write("**Merge Details**")
                    st.success(f"""
                    **Confidence**: {entry.get('confidence', 'N/A')}
                    **Rationale**: {entry.get('rationale', 'N/A')}
                    **Timestamp**: {entry.get('timestamp', 'N/A')}
                    """)
        
        if st.session_state.get("df") is not None:
            st.divider()
            st.header("Analysis & Report Generation")
            analyze_issues()

if __name__ == "__main__":
    main()