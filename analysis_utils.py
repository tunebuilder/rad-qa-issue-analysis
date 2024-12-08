"""
# Analysis utilities for SUDCare QA
# Written by: David Warren
# Last updated: Dec 7, 2024
#
# Core utilities for analyzing QA testing data. Uses OpenAI's GPT-4o 
# to help spot patterns and generate insights from test results.
"""

from typing import Dict, List, Optional
import pandas as pd
from openai import OpenAI
import json
import os
from dotenv import load_dotenv

# Grab env vars
load_dotenv()

# Set up OpenAI - using their official API
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.openai.com/v1"
)

def create_analysis_prompt(df: pd.DataFrame) -> str:
    """
    Builds the prompt sent to GPT-4o for analyzing QA data.
    Takes  test results and formats them in a way the model can understand.
    """
    
    # Get active issues
    active_issues = df[
        (df["Merged With Issue ID"].isna()) |  # Unmerged issues
        (df["Status"].fillna("").str.strip() == "Merged")  # Merged groups
    ]
    
    # Get summary statistics
    total_active = len(active_issues)
    merged_groups = len(active_issues[active_issues["Status"].fillna("").str.strip() == "Merged"])
    unmerged_issues = total_active - merged_groups
    standards = active_issues["Linked Standard"].unique().tolist()
    
    # Get issues grouped by standard
    issues_by_standard = {}
    for standard in standards:
        standard_issues = active_issues[active_issues["Linked Standard"] == standard]
        issues_by_standard[standard] = [
            {
                "issue_id": str(row["Issue ID"]),
                "input_prompt": str(row["Input Prompt"]),
                "failure_rationale": str(row["Failure Rationale"]) if pd.notna(row["Failure Rationale"]) else None,
                "score": float(row["Final Weighted Score (1-3)"]) if pd.notna(row["Final Weighted Score (1-3)"]) else None,
                "status": str(row["Status"]) if pd.notna(row["Status"]) else None,
                "is_merged_group": str(row["Status"]).strip() == "Merged" if pd.notna(row["Status"]) else False  # Handle NA for individual rows
            }
            for _, row in standard_issues.iterrows()
        ]
    
    return f"""Analyze the following QA testing dataset for a mental health support chatbot named Suzy and identify key patterns, priorities, and recommendations.

Dataset Overview:
- Total Active Issues: {total_active}
  * Merged Issue Groups: {merged_groups}
  * Individual Unmerged Issues: {unmerged_issues}
- Standards Evaluated: {len(standards)}

Note: This analysis covers only active issues, which includes merged issue groups and unmerged individual issues.
Individual issues that were merged into groups are not included to avoid redundancy.

Detailed Issues by Standard:
{json.dumps(issues_by_standard, indent=2)}

Please provide your analysis in the following JSON format:

{{
    "summary": {{
        "critical_findings": ["List of 3-5 most critical findings"],
        "overall_assessment": "Brief overall assessment of chatbot performance",
        "dataset_coverage": {{
            "total_active_issues": {total_active},
            "merged_groups": {merged_groups},
            "unmerged_issues": {unmerged_issues},
            "standards_count": {len(standards)}
        }}
    }},
    "standards_analysis": [
        {{
            "standard": "Standard name",
            "total_issues": 123,
            "key_patterns": ["List of identified patterns"],
            "priority_level": "high/medium/low",
            "recommendations": ["List of specific recommendations"]
        }}
    ],
    "priority_areas": [
        {{
            "area": "Description of problem area",
            "affected_standards": ["List of affected standards"],
            "impact": "Description of user/system impact",
            "suggested_fixes": ["List of suggested fixes"],
            "priority_score": 0-100
        }}
    ],
    "improvement_roadmap": [
        {{
            "phase": "1/2/3",
            "focus_area": "Description of focus area",
            "actions": ["List of specific actions"],
            "expected_impact": "Description of expected impact",
            "complexity": "high/medium/low"
        }}
    ]
}}"""

def analyze_qa_issues(df: pd.DataFrame) -> Dict:
    """
    Main analysis function - looks through QA data to find patterns and problems.
    Only looks at active stuff (unmerged issues + merged groups) to avoid duplicates.
    """
    try:
        # Filter for active issues - handle NA values explicitly
        active_issues = df[
            (df["Merged With Issue ID"].isna()) |  # Unmerged issues
            (df["Status"].fillna("").str.strip() == "Merged")  # Merged groups for DataFrame operations
        ].copy()
        
        merged_mask = active_issues["Status"].fillna("").str.strip() == "Merged"
        print(f"Analyzing {len(active_issues)} active issues...")
        print(f"- Merged Groups: {merged_mask.sum()}")
        print(f"- Unmerged Issues: {(~merged_mask).sum()}")
        
        # Create and send the analysis prompt
        prompt = create_analysis_prompt(active_issues)
        
        print("[DEBUG] Sending analysis request to LLM...")
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert at analyzing QA testing data for conversational AI systems.
                    Focus on:
                    1. Identifying systemic issues and patterns
                    2. Prioritizing areas for improvement
                    3. Providing actionable recommendations
                    4. Suggesting concrete steps for implementation
                    
                    Note: The analysis covers only active issues (merged groups and unmerged individuals).
                    Consider merged groups as representing multiple related issues."""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3
        )
        
        # Parse the response
        response_text = completion.choices[0].message.content
        print("\n[DEBUG] Raw LLM response:")
        print(response_text)
        
        # Clean up response if it's in markdown format
        cleaned_response = response_text.strip()
        if cleaned_response.startswith("```"):
            cleaned_response = cleaned_response.split("```")[1]
            if cleaned_response.startswith("json"):
                cleaned_response = cleaned_response[4:]
            cleaned_response = cleaned_response.strip()
        
        # Parse the JSON response
        analysis_results = json.loads(cleaned_response)
        print(f"\n[DEBUG] Successfully parsed analysis results")
        
        return analysis_results
        
    except Exception as e:
        print(f"[DEBUG] Error in analysis: {str(e)}")
        raise Exception(f"Failed to analyze QA issues: {str(e)}")

def calculate_priority_score(issue_count: int, avg_score: float, is_merged: bool) -> float:
    """
    Figures out how important each issue/standard is (0-100 scale).
    
    What affects the score:
    - How many issues we found (more = higher priority)
    - How bad the issues are (higher score = bigger problem)
    - Whether we've seen it before (merged issues = probably systemic)
    """
    # Base score from issue count (max 40 points)
    count_score = min(40, issue_count * 5)
    
    # Score from severity (max 40 points)
    severity_score = min(40, avg_score * 13.33)  # 13.33 * 3 = 40
    
    # Merge status bonus (20 points)
    merge_bonus = 20 if is_merged else 0
    
    return round(count_score + severity_score + merge_bonus, 1)

def generate_priority_areas(df: pd.DataFrame, analysis_results: Dict) -> List[Dict]:
    """
    Takes analysis and raw data to figure out what we need to fix first.
    Spits out a sorted list of problem areas with their priority scores.
    """
    priority_areas = []
    
    # Process each standard
    for standard_analysis in analysis_results["standards_analysis"]:
        standard = standard_analysis["standard"]
        standard_df = df[df["Linked Standard"] == standard]
        
        # Calculate metrics
        issue_count = len(standard_df)
        avg_score = standard_df["Final Weighted Score (1-3)"].mean()
        has_merged = (standard_df["Status"].fillna("").str.strip() == "Merged").any()  # DataFrame operation
        
        # Calculate priority score
        priority_score = calculate_priority_score(issue_count, avg_score, has_merged)
        
        # Add to priority areas if score is significant
        if priority_score > 30:  # Threshold for inclusion
            priority_areas.append({
                "standard": standard,
                "priority_score": priority_score,
                "issue_count": issue_count,
                "avg_severity": round(avg_score, 2),
                "has_merged_issues": has_merged,
                "key_patterns": standard_analysis.get("key_patterns", []),
                "recommendations": standard_analysis.get("recommendations", [])
            })
    
    # Sort by priority score descending
    return sorted(priority_areas, key=lambda x: x["priority_score"], reverse=True)
