"""
PR Strategy Tools with improved file handling.
"""
import os
import logging
import json
from typing import List, Dict, Any, Optional, Union, Tuple
from pydantic import BaseModel, Field, model_validator

from crewai.tools import BaseTool
from shared.models.pr_models import (
    DirectorySummary, 
    GitAnalysisOutput, 
    FileChange
)

logger = logging.getLogger(__name__)

# Create output directory if it doesn't exist
OUTPUT_DIR = "tool_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class SummarizeChangesInput(BaseModel):
    """Input for the summarize changes tool."""
    # Accept ANY dictionary input
    analysis_result: Dict[str, Any] = Field(
        ..., 
        description="Analysis result from GitAnalysiTool"
    )

class ChangeSummary(BaseModel):
    """Summary of code changes for PR grouping."""
    total_files: int = Field(..., description="Total number of changed files")
    directories: List[Dict[str, Any]] = Field(..., description="Changed directories")
    file_extensions: Dict[str, int] = Field(..., description="Count of file extensions")
    largest_changes: List[Dict[str, Any]] = Field(..., description="Files with largest changes")

class SummarizeChangesTool(BaseTool):
    """Tool for summarizing code changes."""
    name: str = "summarize_changes"
    description: str = "Summarize all changes in the repository by directory and file type"
    
    def _run(self, analysis_result: Dict[str, Any]) -> str:
        """Summarize code changes for PR grouping."""
        logger.info("Summarizing changes from analysis")
        
        # Check for cached output
        output_file = os.path.join(OUTPUT_DIR, "summarize_changes_output.json")
        if os.path.exists(output_file):
            logger.info(f"Using cached summary from {output_file}")
            try:
                with open(output_file, 'r') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Error reading cached summary: {e}")
        
        logger.info(f"SummarizeChangesTool received input of type: {type(analysis_result)}")
        logger.info(f"Input keys: {list(analysis_result.keys()) if isinstance(analysis_result, dict) else 'NOT A DICT'}")
        
        try:
            # Create blank summary with default values - ignore validation errors
            # since we can't validate against the actual input format
            
            # We'll build up a minimal ChangeSummary with empty/default data
            total_files = 0
            directories = []
            file_extensions = {}
            largest_changes = []
            
            # Try to extract any useful information from whatever structure we got
            # Approach 1: Look for 'changes' and other fields directly in analysis_result
            if 'changes' in analysis_result and isinstance(analysis_result['changes'], list):
                # Good case - we have changes data directly
                file_changes = analysis_result['changes']
                total_files = len(file_changes)
                
                # Process extensions
                for change in file_changes:
                    ext = "unknown"
                    if isinstance(change, dict) and 'file_path' in change:
                        file_path = change['file_path']
                        if '.' in file_path:
                            ext = file_path.split('.')[-1]
                    file_extensions[ext] = file_extensions.get(ext, 0) + 1
                
                # Get largest changes
                largest_changes = sorted(
                    [
                        {
                            "file_path": c.get('file_path', 'unknown'),
                            "total_changes": c.get('total_changes', 0),
                            "directory": c.get('directory', 'unknown')
                        }
                        for c in file_changes if isinstance(c, dict)
                    ],
                    key=lambda x: x.get("total_changes", 0),
                    reverse=True
                )[:5]  # Limit to top 5
            
            # Approach 2: Look for directory_summaries
            if 'directory_summaries' in analysis_result and isinstance(analysis_result['directory_summaries'], list):
                # Extract directory information
                dir_summaries = analysis_result['directory_summaries']
                for dir_summary in dir_summaries:
                    if isinstance(dir_summary, dict):
                        directories.append({
                            "name": dir_summary.get('name', 'unknown'),
                            "file_count": dir_summary.get('file_count', 0),
                            "files": dir_summary.get('files', [])[:5],
                            "truncated": len(dir_summary.get('files', [])) > 5
                        })
            
            # Approach 3: Look for nested structure with analysis_result field
            elif 'analysis_result' in analysis_result:
                nested_result = analysis_result['analysis_result']
                # Just include basic summary since we can't get actual data
                logger.info("Found nested analysis_result structure")
                
            # Create a summary with whatever information we could extract
            summary = ChangeSummary(
                total_files=total_files,
                directories=directories or [{"name": "(root)", "file_count": 0, "files": [], "truncated": False}],
                file_extensions=file_extensions or {"unknown": 0},
                largest_changes=largest_changes or []
            )
            
            # Convert to JSON
            result_json = summary.model_dump_json(indent=2)
            
            # Save to file
            try:
                with open(output_file, 'w') as f:
                    f.write(result_json)
                logger.info(f"Saved summary to {output_file}")
            except Exception as e:
                logger.error(f"Error saving summary: {e}")
            
            # Return as JSON
            return result_json
            
        except Exception as e:
            logger.exception(f"Error summarizing changes: {e}")
            
            # Return a basic empty summary instead of an error
            # This allows the process to continue even with invalid input
            fallback_summary = ChangeSummary(
                total_files=0,
                directories=[{"name": "(root)", "file_count": 0, "files": [], "truncated": False}],
                file_extensions={"unknown": 0},
                largest_changes=[]
            )
            
            # Include error info in the response
            result = fallback_summary.model_dump()
            result["error"] = str(e)
            result["message"] = "Could not process analysis result, returning empty summary"
            
            result_json = json.dumps(result, indent=2)
            
            # Save error to file
            try:
                with open(output_file, 'w') as f:
                    f.write(result_json)
            except Exception as file_e:
                logger.error(f"Error saving error result: {file_e}")
                
            return result_json


class GetDirectoryDetailsInput(BaseModel):
    """Input for getting directory details."""
    analysis_result: Dict[str, Any] = Field(
        ..., 
        description="Analysis result from GitAnalysisTool"
    )
    directory: str = Field(
        ...,
        description="Directory to get details for"
    )

class GetDirectoryDetailsTool(BaseTool):
    """Tool for getting details of files in a directory."""
    name: str = "get_directory_details"
    description: str = "Get detailed information about files in a specific directory"
    
    def _run(self, analysis_result: Dict[str, Any]) -> str:
        """Summarize code changes for PR grouping."""
        logger.info("Summarizing changes from analysis")
        
        # Check for cached output
        output_file = os.path.join(OUTPUT_DIR, "summarize_changes_output.json")
        if os.path.exists(output_file):
            logger.info(f"Using cached summary from {output_file}")
            try:
                with open(output_file, 'r') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Error reading cached summary: {e}")
        
        # Load any previously saved complete file list
        all_files = self._load_all_files()
        
        try:
            # Extract relevant data from the analysis result
            changes, directory_summaries = self._extract_analysis_data(analysis_result)
            
            # Process in smaller chunks to handle large inputs
            total_files = len(changes)
            logger.info(f"Processing summary for {total_files} changes")
            
            # Process data in manageable batches
            directories = self._process_directories(changes, directory_summaries)
            file_extensions = self._process_file_extensions(changes)
            largest_changes = self._find_largest_changes(changes)
            
            # Create summary with complete file count
            summary = self._create_summary(
                total_files=len(all_files) if all_files else total_files,
                directories=directories,
                file_extensions=file_extensions,
                largest_changes=largest_changes
            )
            
            # Convert to JSON and save
            result_json = json.dumps(summary, indent=2)
            self._save_summary(result_json, output_file)
            
            return result_json
            
        except Exception as e:
            logger.exception(f"Error summarizing changes: {e}")
            error_result = self._create_error_summary(str(e))
            error_json = json.dumps(error_result, indent=2)
            self._save_summary(error_json, output_file)
            return error_json
        
    def _load_all_files(self) -> List[str]:
        """Load the complete list of changed files if available."""
        all_files = []
        try:
            all_files_path = os.path.join(OUTPUT_DIR, "all_changed_files.json")
            if os.path.exists(all_files_path):
                with open(all_files_path, 'r') as f:
                    all_files = json.load(f)
                    logger.info(f"Loaded {len(all_files)} files from complete file list")
        except Exception as e:
            logger.error(f"Error loading complete file list: {e}")
        return all_files

    def _extract_analysis_data(self, analysis_result: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Extract changes and directory summaries from the analysis result."""
        changes = []
        directory_summaries = []
        
        logger.info(f"Input type: {type(analysis_result)}")
        logger.info(f"Input keys: {list(analysis_result.keys()) if isinstance(analysis_result, dict) else 'NOT A DICT'}")
        
        # Approach 1: Direct changes array
        if isinstance(analysis_result, dict):
            if 'changes' in analysis_result and isinstance(analysis_result['changes'], list):
                changes = analysis_result['changes']
                logger.info(f"Found {len(changes)} changes in analysis_result['changes']")
            
            if 'directory_summaries' in analysis_result and isinstance(analysis_result['directory_summaries'], list):
                directory_summaries = analysis_result['directory_summaries']
                logger.info(f"Found {len(directory_summaries)} directory summaries")
        
        # Approach 2: Nested structure
        elif isinstance(analysis_result, dict) and 'analysis_result' in analysis_result:
            nested = analysis_result['analysis_result']
            if isinstance(nested, dict):
                if 'changes' in nested and isinstance(nested['changes'], list):
                    changes = nested['changes']
                    logger.info(f"Found {len(changes)} changes in nested structure")
                
                if 'directory_summaries' in nested and isinstance(nested['directory_summaries'], list):
                    directory_summaries = nested['directory_summaries']
                    logger.info(f"Found {len(directory_summaries)} directory summaries in nested structure")
        
        return changes, directory_summaries

    def _process_directories(self, changes: List[Dict[str, Any]], directory_summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process directory information from changes and summaries."""
        directories = []
        
        # First use any existing directory summaries
        if directory_summaries:
            for dir_summary in directory_summaries:
                if isinstance(dir_summary, dict):
                    directories.append({
                        "name": dir_summary.get('name', 'unknown'),
                        "file_count": dir_summary.get('file_count', 0),
                        "files": dir_summary.get('files', [])[:5],  # Limit to 5 example files
                        "truncated": len(dir_summary.get('files', [])) > 5
                    })
            logger.info(f"Processed {len(directories)} directories from directory summaries")
            return directories
        
        # If no summaries, create directories from changes
        dir_groups = {}
        for change in changes:
            if not isinstance(change, dict):
                continue
                
            # Get directory from file path or use a property
            directory = None
            if 'file_path' in change:
                file_path = change['file_path']
                directory = os.path.dirname(file_path) or "(root)"
            elif 'directory' in change:
                directory = change['directory']
                
            if not directory:
                continue
                
            if directory not in dir_groups:
                dir_groups[directory] = []
                
            if 'file_path' in change:
                dir_groups[directory].append(change['file_path'])
        
        # Convert groups to directory summaries
        for dir_name, files in dir_groups.items():
            directories.append({
                "name": dir_name,
                "file_count": len(files),
                "files": files[:5],  # Limit to 5 example files
                "truncated": len(files) > 5
            })
        
        logger.info(f"Created {len(directories)} directory summaries from changes")
        return directories

    def _process_file_extensions(self, changes: List[Dict[str, Any]]) -> Dict[str, int]:
        """Process file extensions from changes."""
        extensions = {}
        batch_size = 50
        
        # Process in batches to avoid memory issues with large inputs
        for i in range(0, len(changes), batch_size):
            batch = changes[i:min(i + batch_size, len(changes))]
            for change in batch:
                if not isinstance(change, dict):
                    continue
                    
                ext = "unknown"
                if 'extension' in change:
                    ext = change['extension'] or "none"
                elif 'file_path' in change:
                    file_path = change['file_path']
                    if '.' in file_path:
                        ext = file_path.split('.')[-1]
                    else:
                        ext = "none"
                        
                extensions[ext] = extensions.get(ext, 0) + 1
        
        logger.info(f"Found {len(extensions)} different file extensions")
        return extensions

    def _find_largest_changes(self, changes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find the files with the largest changes."""
        change_metrics = []
        
        for change in changes:
            if not isinstance(change, dict):
                continue
                
            file_path = change.get('file_path', 'unknown')
            
            # Calculate total changes
            total_changes = 0
            if 'changes' in change and isinstance(change['changes'], dict):
                total_changes = change['changes'].get('added', 0) + change['changes'].get('deleted', 0)
            elif 'total_changes' in change:
                total_changes = change['total_changes']
            
            # Get directory
            directory = change.get('directory', os.path.dirname(file_path) if file_path != 'unknown' else 'unknown')
            
            change_metrics.append({
                "file_path": file_path,
                "total_changes": total_changes,
                "directory": directory
            })
        
        # Sort by total changes (descending) and get top 10
        largest = sorted(change_metrics, key=lambda x: x["total_changes"], reverse=True)[:10]
        logger.info(f"Identified {len(largest)} largest changes")
        return largest

    def _create_summary(self, total_files: int, directories: List[Dict[str, Any]], 
                    file_extensions: Dict[str, int], largest_changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create the final summary structure."""
        summary = {
            "total_files": total_files,
            "directories": directories or [{"name": "(root)", "file_count": 0, "files": [], "truncated": False}],
            "file_extensions": file_extensions or {"unknown": 0},
            "largest_changes": largest_changes or [],
            "message": f"Processed {total_files} total files across {len(directories)} directories"
        }
        return summary

    def _create_error_summary(self, error_message: str) -> Dict[str, Any]:
        """Create a fallback summary when an error occurs."""
        return {
            "total_files": 0,
            "directories": [{"name": "(root)", "file_count": 0, "files": [], "truncated": False}],
            "file_extensions": {"unknown": 0},
            "largest_changes": [],
            "error": error_message,
            "message": "Could not process analysis result, returning empty summary"
        }

    def _save_summary(self, summary_json: str, output_file: str) -> None:
        """Save summary to output file."""
        try:
            with open(output_file, 'w') as f:
                f.write(summary_json)
            logger.info(f"Saved summary to {output_file}")
        except Exception as e:
            logger.error(f"Error saving summary: {e}")


class CreatePRGroupsInput(BaseModel):
    """Input for creating PR groups."""
    groups: List[Dict[str, Any]] = Field(
        ...,
        description="List of PR groups to create"
    )

class CreatePRGroupsTool(BaseTool):
    """Tool for creating PR groups from agent decisions."""
    name: str = "create_pr_groups"
    description: str = "Create PR groups based on agent's grouping decisions"
    
    def _run(self, groups: Union[List[Dict[str, Any]], Dict[str, Any], str]) -> str:
        """Create PR groups from agent decisions."""
        logger.info(f"Creating PR groups from input of type: {type(groups)}")
        
        # Check for cached output
        output_file = os.path.join(OUTPUT_DIR, "pr_groups_output.json")
        if os.path.exists(output_file):
            logger.info(f"Using cached PR groups from {output_file}")
            try:
                with open(output_file, 'r') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Error reading cached PR groups: {e}")
        
        try:
            # Handle different input formats
            pr_groups_data = []
            
            # Case 1: List of groups
            if isinstance(groups, list):
                pr_groups_data = groups
                
            # Case 2: Dictionary with 'groups' key
            elif isinstance(groups, dict) and 'groups' in groups:
                pr_groups_data = groups['groups']
                
            # Case 3: JSON string
            elif isinstance(groups, str):
                try:
                    parsed_data = json.loads(groups)
                    if isinstance(parsed_data, list):
                        pr_groups_data = parsed_data
                    elif isinstance(parsed_data, dict) and 'groups' in parsed_data:
                        pr_groups_data = parsed_data['groups']
                except json.JSONDecodeError:
                    logger.error(f"Could not parse input as JSON: {groups[:100]}...")
                    pr_groups_data = []
            
            logger.info(f"Processing {len(pr_groups_data)} PR groups")
            
            # Create PR groups
            pr_groups = []
            for group_data in pr_groups_data:
                # Extract fields with defaults
                title = group_data.get('title', 'Untitled PR')
                files = group_data.get('files', [])
                rationale = group_data.get('rationale', 'No rationale provided')
                suggested_branch = group_data.get('suggested_branch', 'unnamed-branch')
                description = group_data.get('description')
                
                # Sanitize data - ensure files is a list
                if not isinstance(files, list):
                    files = []
                
                # Create PR group dict
                pr_group = {
                    "title": title,
                    "files": files,
                    "rationale": rationale,
                    "suggested_branch": suggested_branch,
                    "description": description
                }
                pr_groups.append(pr_group)
            
            # Create PR suggestion
            suggestion = {
                "pr_suggestions": pr_groups,
                "total_groups": len(pr_groups),
                "description": f"Generated {len(pr_groups)} PR suggestions",
                "message": f"Generated {len(pr_groups)} PR suggestions"
            }
            
            result_json = json.dumps(suggestion, indent=2)
            
            # Save to file
            try:
                with open(output_file, 'w') as f:
                    f.write(result_json)
                logger.info(f"Saved PR groups to {output_file}")
            except Exception as e:
                logger.error(f"Error saving PR groups: {e}")
            
            return result_json
            
        except Exception as e:
            logger.exception(f"Error creating PR groups: {e}")
            
            # Return minimal valid response
            error_result = {
                "pr_suggestions": [],
                "total_groups": 0,
                "error": str(e),
                "message": "Failed to create PR groups"
            }
            
            error_json = json.dumps(error_result, indent=2)
            
            # Save error to file
            try:
                with open(output_file, 'w') as f:
                    f.write(error_json)
            except Exception as file_e:
                logger.error(f"Error saving error result: {file_e}")
                
            return error_json