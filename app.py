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

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize merge executor
merge_executor = MergeExecutor()

# Configure Streamlit page
st.set_page_config(
    page_title="QA Issues Analysis Tool",
    page_icon="📊",
    layout="wide"
)

# Define the QA standards
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

# Initialize session state variables if they don't exist
if 'df' not in st.session_state:
    st.session_state.df = None
if 'merge_suggestions' not in st.session_state:
    st.session_state.merge_suggestions = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = 0

# Add custom CSS
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
    """Load and validate the uploaded CSV file"""
    try:
        df = pd.read_csv(uploaded_file)
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
            
        return df, None
    except Exception as e:
        return None, f"Error loading file: {str(e)}"

def display_merge_preview(df: pd.DataFrame, merge_suggestion: dict):
    """Display a detailed preview of a merge operation"""
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
    
    # Display secondary issues
    st.write("**Issues to be Merged**")
    for issue_id in secondary_issues:
        issue_data = df[df["Issue ID"] == issue_id].iloc[0]
        st.warning(f"""
        **ID**: {issue_id}
        **Input**: {issue_data['Input Prompt']}
        **Score**: {issue_data['Final Weighted Score (1-3)']}
        **Rationale**: {issue_data['Failure Rationale']}
        """)
    
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

        # Display issue counts
        active_issues = df[
            (df["Merged With Issue ID"].isna()) |  # Unmerged issues
            (df["Status"] == "Merged")  # Merged groups
        ]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Active Issues", len(active_issues))
        with col2:
            st.metric("Merged Groups", len(df[df["Status"] == "Merged"]))
        with col3:
            st.metric("Unmerged Issues", len(df[df["Status"].isna()]))

        # Show analysis scope
        st.info("""
        ℹ️ Analysis Scope:
        
        The analysis will be performed on active issues only, which includes:
        1. Merged issue groups (representing multiple related issues)
        2. Individual unmerged issues
        
        Individual issues that were merged into groups are excluded to avoid redundancy.
        """)

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
                active_issues = len(df[df["Status"].isna()])
                st.metric("Active Issues", active_issues)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                merged_issues = len(df[df["Status"] == "Merged"])
                st.metric("Merged Issues", merged_issues)
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
                    
                    with st.expander(f"View Details"):
                        display_merge_preview(df, suggestion)
                        
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
            
            # Get cache status
            use_cache = st.session_state.get('use_cache', True)
            merge_history = merge_executor.auditor.get_merge_history(use_cache=use_cache)
            
            if not merge_history:
                if not use_cache:
                    st.info("Cache usage is disabled. Enable it in the Merge Analysis tab to view merge history.")
                else:
                    st.info("No merge history available yet.")
            else:
                for entry in merge_history:
                    with st.expander(f"Merge: {entry['primary_issue']} + {len(entry['merged_issues'])} issues"):
                        st.write(f"**Primary Issue:** {entry['primary_issue']}")
                        st.write(f"**Merged Issues:** {', '.join(entry['merged_issues'])}")
                        st.write(f"**Confidence:** {entry['confidence']:.2f}")
                        st.write(f"**Rationale:** {entry['rationale']}")
                        st.write(f"**Timestamp:** {entry['timestamp']}")
        
        if st.session_state.get("df") is not None:
            st.divider()
            st.header("Analysis & Report Generation")
            analyze_issues()

if __name__ == "__main__":
    main()