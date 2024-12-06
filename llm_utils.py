from openai import OpenAI
import pandas as pd
from typing import List, Dict, Tuple
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

print(f"[DEBUG] Using API key starting with: {api_key[:10]}...")

# Initialize OpenAI client
client = OpenAI(
    api_key=api_key,
    base_url="https://api.openai.com/v1"  # Explicitly set base URL
)

def create_merge_analysis_prompt(issues: List[Dict]) -> str:
    """Create a prompt for analyzing potential issue merges"""
    return f"""Analyze the following QA issues and identify which ones should be merged based on similar root causes or overlapping problems.
For each potential merge, explain the rationale and provide a confidence score (0-1).

Issues to analyze:
{json.dumps(issues, indent=2)}

Provide your response in the following JSON format:
{{
    "merge_groups": [
        {{
            "issues": ["issue_id1", "issue_id2"],
            "rationale": "Explanation for why these should be merged",
            "confidence": 0.95
        }}
    ]
}}"""

def analyze_issues_for_merge(df: pd.DataFrame, batch_size: int = 50) -> List[Dict]:
    """
    Analyze issues to identify potential merges using the LLM.
    Returns a list of merge suggestions.
    """
    all_merge_suggestions = []
    
    # Filter for open issues only
    open_issues_df = df[df["Status"].fillna("Open") == "Open"]
    print(f"\n[DEBUG] Found {len(open_issues_df)} open issues out of {len(df)} total issues")
    
    if len(open_issues_df) == 0:
        print("[DEBUG] No open issues found to analyze")
        return []
    
    # Group issues by standard first
    standards = open_issues_df["Linked Standard"].unique()
    print(f"[DEBUG] Found {len(standards)} unique standards to analyze")
    
    for standard in standards:
        # Get issues for this standard
        standard_df = open_issues_df[open_issues_df["Linked Standard"] == standard]
        print(f"\n[DEBUG] Processing standard: {standard}")
        print(f"[DEBUG] Found {len(standard_df)} open issues for this standard")
        
        # Skip if less than 2 issues for this standard
        if len(standard_df) < 2:
            print("[DEBUG] Skipping standard - less than 2 issues")
            continue
            
        # Process issues in batches to stay within token limits
        for i in range(0, len(standard_df), batch_size):
            batch = standard_df.iloc[i:i+batch_size]
            print(f"\n[DEBUG] Processing batch {i//batch_size + 1} with {len(batch)} issues")
            
            # Create a list of issue dictionaries for the batch
            issues = []
            for _, row in batch.iterrows():
                issue = {
                    "issue_id": row["Issue ID"],
                    "input_prompt": row["Input Prompt"],
                    "failure_rationale": row["Failure Rationale"],
                    "linked_standard": row["Linked Standard"].strip(),
                    "final_score": row["Final Weighted Score (1-3)"]
                }
                issues.append(issue)
                print(f"[DEBUG] Added issue {issue['issue_id']} to batch")
            
            # Skip if not enough issues in batch
            if len(issues) < 2:
                print("[DEBUG] Skipping batch - less than 2 issues")
                continue
            
            # Get merge suggestions from LLM
            try:
                print("\n[DEBUG] Sending request to LLM...")
                completion = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": """You are an expert at analyzing QA issues and identifying patterns and similarities between issues.
                            When analyzing issues:
                            1. Look for similar root causes or overlapping problems
                            2. Issues with similar themes or patterns should be merged
                            3. Provide detailed rationale for suggested merges"""
                        },
                        {
                            "role": "user",
                            "content": create_merge_analysis_prompt(issues)
                        }
                    ],
                    temperature=0.3  # Lower temperature for more consistent analysis
                )
                
                # Parse the response
                response_text = completion.choices[0].message.content
                print("\n[DEBUG] Raw LLM response:")
                print(response_text)
                print("\n[DEBUG] Attempting to parse JSON response...")
                
                try:
                    # Clean up the response text to handle markdown formatting
                    cleaned_response = response_text.strip()
                    if cleaned_response.startswith("```"):
                        # Remove markdown code block formatting
                        cleaned_response = cleaned_response.split("```")[1]  # Get content between first set of ```
                        if cleaned_response.startswith("json"):
                            cleaned_response = cleaned_response[4:]  # Remove "json" language identifier
                        cleaned_response = cleaned_response.strip()
                    
                    print("[DEBUG] Cleaned response:")
                    print(cleaned_response)
                    
                    merge_suggestions = json.loads(cleaned_response)
                    new_suggestions = merge_suggestions.get("merge_groups", [])
                    print(f"[DEBUG] Successfully parsed {len(new_suggestions)} merge suggestions")
                    for suggestion in new_suggestions:
                        print(f"\n[DEBUG] Merge suggestion:")
                        print(f"  Issues: {suggestion['issues']}")
                        print(f"  Confidence: {suggestion['confidence']}")
                        print(f"  Rationale: {suggestion['rationale']}")
                    all_merge_suggestions.extend(new_suggestions)
                    
                except json.JSONDecodeError as e:
                    print(f"[DEBUG] Error parsing LLM response: {str(e)}")
                    print("[DEBUG] Invalid JSON structure in response")
                    continue
                
            except Exception as e:
                print(f"[DEBUG] Error processing batch: {str(e)}")
                continue
    
    print(f"\n[DEBUG] Analysis complete")
    print(f"[DEBUG] Found {len(all_merge_suggestions)} total merge suggestions across {len(standards)} standards")
    return all_merge_suggestions

def apply_merges(df: pd.DataFrame, merge_suggestions: List[Dict]) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Apply the suggested merges to the DataFrame and return the updated DataFrame
    along with a list of merge actions taken.
    """
    merge_actions = []
    df = df.copy()
    
    for suggestion in merge_suggestions:
        if suggestion["confidence"] >= 0.8:  # Only apply high-confidence merges
            issues = suggestion["issues"]
            if len(issues) >= 2:
                primary_issue = issues[0]
                secondary_issues = issues[1:]
                
                # Record the merge action
                merge_action = {
                    "primary_issue": primary_issue,
                    "merged_issues": secondary_issues,
                    "rationale": suggestion["rationale"],
                    "confidence": suggestion["confidence"]
                }
                merge_actions.append(merge_action)
                
                # Update the Merged IDs field for the primary issue
                merged_ids = df.loc[df["Issue ID"] == primary_issue, "Merged IDs"].iloc[0]
                if pd.isna(merged_ids):
                    merged_ids = ""
                new_merged_ids = (merged_ids + "," if merged_ids else "") + ",".join(secondary_issues)
                df.loc[df["Issue ID"] == primary_issue, "Merged IDs"] = new_merged_ids
                
                # Mark secondary issues as merged
                df.loc[df["Issue ID"].isin(secondary_issues), "Status"] = "Merged"
                df.loc[df["Issue ID"].isin(secondary_issues), "Merged With Issue ID"] = primary_issue
    
    return df, merge_actions

# Explicitly export the functions
__all__ = ['create_merge_analysis_prompt', 'analyze_issues_for_merge', 'apply_merges']
