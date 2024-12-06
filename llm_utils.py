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

def analyze_issues_for_merge(df: pd.DataFrame) -> List[Dict]:
    """
    Analyze issues to identify potential merges using the LLM.
    Returns a list of merge suggestions.
    """
    all_merge_suggestions = []
    processed_issues = set()  # Track which issues have been included in suggestions
    
    # Filter for unmerged issues only
    unmerged_mask = (
        pd.isna(df["Status"]) & 
        pd.isna(df["Merged With Issue ID"]) & 
        pd.isna(df["Merged IDs"])
    )
    open_issues_df = df[unmerged_mask].copy()
    
    print(f"\n[DEBUG] Status value counts before filtering:\n{df['Status'].value_counts(dropna=False)}")
    print(f"[DEBUG] Found {len(open_issues_df)} unmerged issues out of {len(df)} total issues")
    
    if len(open_issues_df) == 0:
        print("[DEBUG] No unmerged issues found to analyze")
        return []
    
    # Group issues by standard first
    standards = open_issues_df["Linked Standard"].unique()
    print(f"[DEBUG] Found {len(standards)} unique standards to analyze")
    
    for standard in standards:
        # Get all issues for this standard
        standard_df = open_issues_df[open_issues_df["Linked Standard"] == standard]
        print(f"\n[DEBUG] Processing standard: {standard}")
        print(f"[DEBUG] Found {len(standard_df)} unmerged issues for this standard")
        
        # Skip if less than 2 issues for this standard
        if len(standard_df) < 2:
            print("[DEBUG] Skipping standard - less than 2 issues")
            continue
        
        # Create list of all unprocessed issues for this standard
        standard_issues = []
        for _, row in standard_df.iterrows():
            issue_id = row["Issue ID"]
            # Skip if this issue has already been processed
            if issue_id in processed_issues:
                continue
                
            issue = {
                "issue_id": issue_id,
                "input_prompt": row["Input Prompt"],
                "failure_rationale": row["Failure Rationale"],
                "linked_standard": row["Linked Standard"].strip(),
                "final_score": row["Final Weighted Score (1-3)"]
            }
            standard_issues.append(issue)
        
        # Skip if no unprocessed issues
        if len(standard_issues) < 2:
            print("[DEBUG] No unprocessed issues for this standard")
            continue
            
        print(f"[DEBUG] Analyzing {len(standard_issues)} unprocessed issues for standard {standard}")
        
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
                        3. Consider all possible relationships between issues
                        4. Group related issues together - don't split them across multiple suggestions
                        5. Only suggest merges when there is strong similarity (confidence >= 0.8)
                        6. You can suggest multiple issues be merged together if they are all related"""
                    },
                    {
                        "role": "user",
                        "content": create_merge_analysis_prompt(standard_issues)
                    }
                ],
                temperature=0.3  # Lower temperature for more consistent analysis
            )
            
            # Parse the response
            response_text = completion.choices[0].message.content
            print("\n[DEBUG] Raw LLM response:")
            print(response_text)
            
            try:
                # Clean up the response text to handle markdown formatting
                cleaned_response = response_text.strip()
                if cleaned_response.startswith("```"):
                    cleaned_response = cleaned_response.split("```")[1]
                    if cleaned_response.startswith("json"):
                        cleaned_response = cleaned_response[4:]
                    cleaned_response = cleaned_response.strip()
                
                merge_suggestions = json.loads(cleaned_response)
                new_suggestions = merge_suggestions.get("merge_groups", [])
                
                # Filter out suggestions that include already processed issues
                valid_suggestions = []
                for suggestion in new_suggestions:
                    issues = suggestion["issues"]
                    if not any(issue in processed_issues for issue in issues):
                        # Only accept high confidence suggestions
                        if suggestion["confidence"] >= 0.8:
                            valid_suggestions.append(suggestion)
                            # Mark these issues as processed
                            processed_issues.update(issues)
                
                print(f"[DEBUG] Found {len(valid_suggestions)} valid merge suggestions")
                all_merge_suggestions.extend(valid_suggestions)
                
            except json.JSONDecodeError as e:
                print(f"[DEBUG] Error parsing LLM response: {str(e)}")
                continue
            
        except Exception as e:
            print(f"[DEBUG] Error processing standard {standard}: {str(e)}")
            continue
    
    print(f"\n[DEBUG] Analysis complete")
    print(f"[DEBUG] Found {len(all_merge_suggestions)} total merge suggestions")
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
