"""
Tools for validating and improving PR suggestions.
"""
from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel, Field
import logging
import json
import os
from crewai.tools import BaseTool
from shared.models.pr_models import PRSuggestion, PullRequestGroup

logger = logging.getLogger(__name__)

class ValidationSuggestion(BaseModel):
    """Structured suggestion for PR improvement."""
    type: str = Field(..., description="Type of suggestion (missing_files, balance, etc.)")
    message: str = Field(..., description="Human-readable message about the suggestion")
    details: str = Field(..., description="Detailed information for improvement")

class ValidationResult(BaseModel):
    """Pydantic model for validation results."""
    validation_status: str = Field(..., description="Pass or fail status of validation")
    total_changed_files: int = Field(..., description="Total number of changed files")
    files_in_prs: int = Field(..., description="Number of files included in PRs")
    missing_files: List[str] = Field(default_factory=list, description="Files not in any PR")
    missing_file_count: int = Field(..., description="Count of missing files")
    pr_group_count: int = Field(..., description="Number of PR groups")
    pr_group_sizes: List[int] = Field(default_factory=list, description="Sizes of each PR group")
    balanced_groups: bool = Field(..., description="Whether PR groups are relatively balanced")
    suggestions: List[ValidationSuggestion] = Field(default_factory=list, description="Suggestions for improvement")

class ValidationTool(BaseTool):
    """Tool for validating PR suggestions against original analysis."""
    name: str = "validate_pr_suggestions"
    description: str = "Validate that PR suggestions include all changed files and follow best practices"
    
    def __init__(self):
        super().__init__()
    
    def _run(self, original_analysis: Dict[str, Any], pr_suggestions: Dict[str, Any]) -> str:
        """Validate PR suggestions against original file analysis."""
        logger.info("Validating PR suggestions")
        
        try:
            # First check for the all_files list
            all_files = set()
            try:
                if os.path.exists('all_changed_files.json'):
                    with open('all_changed_files.json', 'r') as f:
                        all_files = set(json.load(f))
                        logger.info(f"Loaded {len(all_files)} files from all_changed_files.json")
            except Exception as e:
                logger.error(f"Error loading all files: {e}")
            
            # If all_files is empty, try to extract from original_analysis
            if not all_files:
                if 'changes' in original_analysis and isinstance(original_analysis['changes'], list):
                    for change in original_analysis['changes']:
                        if isinstance(change, dict) and 'file_path' in change:
                            all_files.add(change['file_path'])
            
            # Extract files included in PR suggestions
            included_files = set()
            if 'pr_suggestions' in pr_suggestions:
                for pr in pr_suggestions['pr_suggestions']:
                    if isinstance(pr, dict) and 'files' in pr:
                        included_files.update(pr.get('files', []))
            
            # Find missing files
            missing_files = all_files - included_files
            
            # Analyze PR groups for balance
            pr_groups = pr_suggestions.get('pr_suggestions', [])
            group_sizes = [len(pr.get('files', [])) for pr in pr_groups]
            
            # Create validation result with missing files
            result = ValidationResult(
                validation_status="pass" if not missing_files else "fail",
                total_changed_files=len(all_files),
                files_in_prs=len(included_files),
                missing_files=list(missing_files),
                missing_file_count=len(missing_files),
                pr_group_count=len(pr_groups),
                pr_group_sizes=group_sizes,
                balanced_groups=max(group_sizes or [0]) <= min(group_sizes or [1]) * 3 if group_sizes else True,
                suggestions=[]
            )
            
            # Add missing files to the result
            pr_suggestions["missing_files"] = list(missing_files)
            pr_suggestions["validation_result"] = result.model_dump()
            
            # Return the PR suggestions with validation info
            return json.dumps(pr_suggestions, indent=2)
        
        except Exception as e:
            logger.exception(f"Error validating PR suggestions: {e}")
            return json.dumps({"error": str(e)})

class PRRebalancer(BaseTool):
    """Tool for rebalancing PR groups to better distribute files."""
    name: str = "rebalance_pr_groups"
    description: str = "Rebalance PR groups to distribute files more evenly"
    
    def _run(self, pr_suggestions: Dict[str, Any], validation_result: Dict[str, Any]) -> str:
        """Rebalance PR groups and add missing files."""
        logger.info("Rebalancing PR groups")
        
        try:
            # Get missing files from validation_result
            missing_files = validation_result.get('missing_files', [])
            if not missing_files and 'missing_files' in pr_suggestions:
                missing_files = pr_suggestions.get('missing_files', [])
            
            logger.info(f"Found {len(missing_files)} missing files to add")
            
            if missing_files and 'pr_suggestions' in pr_suggestions:
                pr_groups = pr_suggestions['pr_suggestions']
                
                # Group missing files by directory
                files_by_dir = {}
                for file in missing_files:
                    dir_name = os.path.dirname(file) or '(root)'
                    if dir_name not in files_by_dir:
                        files_by_dir[dir_name] = []
                    files_by_dir[dir_name].append(file)
                
                # For each directory group, find the best PR or create a new one
                for dir_name, files in files_by_dir.items():
                    # Try to find a PR with files from the same directory
                    found_pr = False
                    for pr in pr_groups:
                        pr_dirs = set()
                        for f in pr.get('files', []):
                            pr_dirs.add(os.path.dirname(f) or '(root)')
                        
                        # If this PR has files from the same directory, add to it
                        if dir_name in pr_dirs:
                            pr['files'].extend(files)
                            found_pr = True
                            logger.info(f"Added {len(files)} files from {dir_name} to existing PR")
                            break
                    
                    # If no suitable PR found, create a new one
                    if not found_pr:
                        new_pr = {
                            "title": f"Additional Changes in {dir_name}",
                            "files": files,
                            "rationale": f"Grouping additional files from {dir_name}",
                            "suggested_branch": f"additional-{dir_name.replace('/', '-')}",
                            "description": f"Additional changes in {dir_name} that weren't included in previous PRs"
                        }
                        pr_groups.append(new_pr)
                        logger.info(f"Created new PR for {len(files)} files from {dir_name}")
                
                # Update PR suggestions
                pr_suggestions['pr_suggestions'] = pr_groups
                pr_suggestions['message'] = f"Rebalanced PR suggestions to include {len(missing_files)} previously missing files"
                pr_suggestions['total_groups'] = len(pr_groups)
            
            return json.dumps(pr_suggestions, indent=2)
        except Exception as e:
            logger.exception(f"Error rebalancing PR groups: {e}")
            return json.dumps({"error": str(e)})