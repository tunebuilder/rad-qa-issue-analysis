from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime
import json
import os

"""
# QA Issue Merge Utilities
# Author: David Warren
# Created: Dec 7, 2024
#
# Core classes for handling our QA issue merges.
# Includes validation, auditing, and the actual merge logic.
"""

class MergeValidator:
    """Makes sure we don't mess up our data when merging issues"""
    
    @staticmethod
    def validate_merge_group(df: pd.DataFrame, issues: List[str]) -> Tuple[bool, str]:
        """
        Checks if we can safely merge a group of issues.
        Returns (ok_to_merge, error_msg).
        """
        if len(issues) < 2:
            return False, "Need at least 2 issues to merge"
            
        # Check if all issues exist
        existing_issues = df["Issue ID"].tolist()
        missing_issues = [issue for issue in issues if issue not in existing_issues]
        if missing_issues:
            return False, f"Issues not found: {', '.join(missing_issues)}"
            
        # Check if any issues are already merged
        merged_issues = df[df["Status"] == "Merged"]["Issue ID"].tolist()
        already_merged = [issue for issue in issues if issue in merged_issues]
        if already_merged:
            return False, f"Issues already merged: {', '.join(already_merged)}"
            
        # Check if issues share the same standard
        standards = df[df["Issue ID"].isin(issues)]["Linked Standard"].unique()
        if len(standards) > 1:
            return False, f"Issues have different standards: {', '.join(standards)}"
            
        return True, ""

class MergeAuditor:
    """Keeps track of what we merged and when"""
    
    def __init__(self, audit_file: str = "merge_audit.jsonl"):
        self.audit_file = audit_file
        
    def log_merge(self, merge_action: Dict) -> None:
        """Writes merge details to our audit log"""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": "merge",
            **merge_action
        }
        
        with open(self.audit_file, "a") as f:
            f.write(json.dumps(audit_entry) + "\n")
    
    def get_merge_history(self, use_cache: bool = True) -> List[Dict]:
        """Pulls up our merge history from the log file"""
        if not use_cache:
            return []
            
        try:
            with open(self.audit_file, "r") as f:
                return [json.loads(line) for line in f]
        except FileNotFoundError:
            return []
            
    def clear_cache(self) -> bool:
        """Wipes out the merge history if needed"""
        try:
            if os.path.exists(self.audit_file):
                os.remove(self.audit_file)
                return True
            return False
        except Exception:
            return False

    def get_unmerged_issues_count(self, df: pd.DataFrame) -> int:
        """
        Counts how many issues haven't been merged yet.
        An issue is still unmerged if it's not:
        - Part of a merge group
        - A secondary issue in someone else's merge
        - The primary issue in a merge
        """
        return len(df[
            pd.isna(df["Status"]) & 
            pd.isna(df["Merged With Issue ID"]) & 
            pd.isna(df["Merged IDs"])
        ])

class MergeExecutor:
    """Handles the actual merge operations with safety checks"""
    
    def __init__(self):
        self.validator = MergeValidator()
        self.auditor = MergeAuditor()
        
    def combine_field_values(self, values: List[str], field: str) -> str:
        """
        Figures out how to combine values from different issues.
        Each field type needs its own rules for combining.
        """
        # Remove empty values and duplicates
        values = [str(v) for v in values if pd.notna(v) and str(v).strip()]
        values = list(dict.fromkeys(values))
        
        if not values:
            return ""
            
        if field == "Final Weighted Score (1-3)":
            # Use the highest score
            return max(float(v) for v in values)
        elif field == "Failure Rationale":
            # Combine rationales with bullets
            return "\n".join(f"• {v}" for v in values)
        elif field == "Investigation Notes":
            # Combine notes with timestamps
            return "\n\n".join(f"[Previous Note] {v}" for v in values)
        else:
            # Default: use the first non-empty value
            return values[0]
    
    def execute_merge(self, df: pd.DataFrame, merge_suggestion: Dict) -> Tuple[pd.DataFrame, Optional[Dict]]:
        """
        Does the actual merge after running safety checks.
        Returns the updated data and what we did, or None if something went wrong.
        """
        # Get the selected issues (if available) or use all issues
        issues = merge_suggestion.get("selected_issues", merge_suggestion["issues"])
        
        # Validate the merge group
        is_valid, error = self.validator.validate_merge_group(df, issues)
        if not is_valid:
            print(f"[ERROR] Merge validation failed: {error}")
            return df, None
            
        # Extract issue information
        primary_issue = issues[0]
        secondary_issues = issues[1:]
        
        # Create a copy of the DataFrame
        df = df.copy()
        
        # Update primary issue
        df.loc[df["Issue ID"] == primary_issue, "Status"] = "Primary"
        
        # Get all field values for combining
        all_values = {
            field: df[df["Issue ID"].isin(issues)][field].tolist()
            for field in ["Input Prompt", "Failure Rationale"]
        }
        
        # Combine field values
        combined_values = {
            field: self.combine_field_values(values, field)
            for field, values in all_values.items()
        }
        
        # Update primary issue with combined values
        for field, value in combined_values.items():
            df.loc[df["Issue ID"] == primary_issue, field] = value
            
        # Track merged IDs in primary issue
        df.loc[df["Issue ID"] == primary_issue, "Merged IDs"] = json.dumps(secondary_issues)
        
        # Update secondary issues
        for issue_id in secondary_issues:
            df.loc[df["Issue ID"] == issue_id, "Status"] = "Merged"
            df.loc[df["Issue ID"] == issue_id, "Merged With Issue ID"] = primary_issue
            
        # Create merge action for audit
        merge_action = {
            "primary_issue": primary_issue,
            "secondary_issues": secondary_issues,
            "confidence": merge_suggestion["confidence"],
            "rationale": merge_suggestion["rationale"]
        }
        
        # Log the merge action
        self.auditor.log_merge(merge_action)
        
        return df, merge_action
