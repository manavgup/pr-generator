# mcp_servers/pr_analyzer/services/workflow_service.py
"""Workflow service for orchestrating complete PR generation workflows."""
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

from mcp_servers.pr_analyzer.services.analysis_service import AnalysisService
from mcp_servers.pr_analyzer.services.grouping_service import GroupingService
from mcp_servers.pr_analyzer.services.validation_service import ValidationService

logger = logging.getLogger(__name__)

class WorkflowService:
    """Service for orchestrating complete workflows."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.analysis_service = AnalysisService(config)
        self.grouping_service = GroupingService(config)
        self.validation_service = ValidationService(config)
    
    async def run_complete_workflow(
        self,
        repo_path: str,
        strategy: Optional[str] = None,
        max_files_per_pr: int = 30,
        target_batch_size: int = 50,
        validate: bool = True,
        generate_metadata: bool = True
    ) -> Dict[str, Any]:
        """Run the complete PR generation workflow."""
        workflow_id = f"workflow_{datetime.now().isoformat()}"
        
        try:
            logger.info(f"Starting workflow {workflow_id} for {repo_path}")
            
            # Step 1: Analyze repository
            analysis = await self.analysis_service.analyze_repository(
                repo_path=repo_path,
                include_stats=True
            )
            
            # Step 2: Suggest PR boundaries
            boundaries = await self.grouping_service.suggest_pr_boundaries(
                repository_analysis=analysis,
                strategy=strategy,
                max_files_per_pr=max_files_per_pr,
                target_batch_size=target_batch_size
            )
            
            # Step 3: Validate groups (optional)
            validation = None
            if validate:
                validation = await self.validation_service.validate_groups(
                    pr_grouping_strategy=boundaries,
                    is_final_validation=True
                )
                
                # If validation failed, refine the groups
                if not validation.get("is_valid", True):
                    boundaries = await self.validation_service.refine_groups(
                        pr_grouping_strategy=boundaries,
                        validation_result=validation,
                        original_repository_analysis=analysis
                    )
            
            # Step 4: Generate metadata for each group (optional)
            if generate_metadata and "groups" in boundaries:
                enhanced_groups = []
                for group in boundaries["groups"]:
                    enhanced_group = await self.generate_pr_metadata(
                        pr_group=group,
                        repository_analysis=analysis
                    )
                    enhanced_groups.append(enhanced_group)
                boundaries["groups"] = enhanced_groups
            
            return {
                "status": "success",
                "workflow_id": workflow_id,
                "repository": repo_path,
                "analysis_summary": {
                    "total_files": analysis.get("total_files_changed", 0),
                    "total_lines": analysis.get("total_lines_changed", 0),
                    "file_types": len(analysis.get("extensions_summary", {}))
                },
                "pr_suggestions": boundaries,
                "validation": validation,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in workflow {workflow_id}: {e}", exc_info=True)
            return {
                "status": "error",
                "workflow_id": workflow_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def generate_pr_metadata(
        self,
        pr_group: Dict[str, Any],
        template: str = "standard",
        repository_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate PR metadata including title, description, and labels."""
        try:
            # Extract basic info
            title = pr_group.get("title", "Update files")
            files = pr_group.get("files", [])
            rationale = pr_group.get("rationale", "")
            
            # Generate branch name
            branch_name = self._generate_branch_name(title)
            
            # Generate labels
            labels = self._generate_labels(files, pr_group)
            
            # Generate description
            description = self._generate_description(
                pr_group=pr_group,
                template=template,
                repository_analysis=repository_analysis
            )
            
            # Enhance the original group with metadata
            enhanced_group = pr_group.copy()
            enhanced_group.update({
                "branch": branch_name,
                "labels": labels,
                "description": description,
                "metadata": {
                    "template": template,
                    "generated_at": datetime.now().isoformat(),
                    "file_count": len(files)
                }
            })
            
            return enhanced_group
            
        except Exception as e:
            logger.error(f"Error generating PR metadata: {e}", exc_info=True)
            raise
    
    async def export_pr_groups(
        self,
        pr_grouping_strategy: Dict[str, Any],
        format: str = "json",
        include_diffs: bool = False
    ) -> Dict[str, Any]:
        """Export PR groups in various formats."""
        try:
            if format == "json":
                return pr_grouping_strategy
            elif format == "markdown":
                return self._export_as_markdown(pr_grouping_strategy)
            elif format == "csv":
                return self._export_as_csv(pr_grouping_strategy)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Error exporting PR groups: {e}", exc_info=True)
            raise
    
    def _generate_branch_name(self, title: str) -> str:
        """Generate a valid git branch name from title."""
        import re
        
        # Convert to lowercase and replace spaces/special chars
        branch = title.lower()
        branch = re.sub(r'[^a-z0-9-]', '-', branch)
        branch = re.sub(r'-+', '-', branch)
        branch = branch.strip('-')
        
        # Ensure it's not empty and limit length
        if not branch:
            branch = 'update'
        branch = branch[:50]
        
        return f"feature/{branch}"
    
    def _generate_labels(self, files: List[str], pr_group: Dict[str, Any]) -> List[str]:
        """Generate PR labels based on files and metadata."""
        labels = ["auto-generated"]
        
        # Add labels based on file types
        extensions = set()
        for file in files:
            if '.' in file:
                ext = file.split('.')[-1]
                extensions.add(ext)
        
        if 'py' in extensions:
            labels.append("python")
        if 'js' in extensions or 'ts' in extensions:
            labels.append("javascript")
        if 'md' in extensions:
            labels.append("documentation")
        
        # Add labels based on directories
        if pr_group.get("directory_focus"):
            labels.append(f"area:{pr_group['directory_focus']}")
        
        # Add feature label if available
        if pr_group.get("feature_focus"):
            labels.append(f"feature:{pr_group['feature_focus']}")
        
        return labels
    
    def _generate_description(
        self,
        pr_group: Dict[str, Any],
        template: str,
        repository_analysis: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate PR description based on template."""
        # Get template content (would normally fetch from templates resource)
        template_content = """## Description
{description}

## Changes
{changes}

## Rationale
{rationale}

## Impact
- Files changed: {file_count}
- Estimated complexity: {complexity}
"""
        
        # Prepare variables
        files = pr_group.get("files", [])
        changes_list = "\n".join([f"- `{file}`" for file in files[:10]])
        if len(files) > 10:
            changes_list += f"\n- ... and {len(files) - 10} more files"
        
        # Fill template
        description = template_content.format(
            description=pr_group.get("title", "Update files"),
            changes=changes_list,
            rationale=pr_group.get("rationale", "Group of related changes"),
            file_count=len(files),
            complexity=pr_group.get("estimated_size", "Medium")
        )
        
        return description
    
    def _export_as_markdown(self, pr_grouping_strategy: Dict[str, Any]) -> Dict[str, Any]:
       """Export PR groups as markdown."""
       markdown_content = []
       
       # Header
       markdown_content.append("# PR Groups Export")
       markdown_content.append(f"\nGenerated at: {datetime.now().isoformat()}")
       markdown_content.append(f"Strategy: {pr_grouping_strategy.get('strategy_type', 'Unknown')}\n")
       
       # Summary
       groups = pr_grouping_strategy.get('groups', [])
       markdown_content.append(f"## Summary")
       markdown_content.append(f"- Total PRs: {len(groups)}")
       markdown_content.append(f"- Total files: {sum(len(g.get('files', [])) for g in groups)}")
       markdown_content.append(f"- Review complexity: {pr_grouping_strategy.get('estimated_review_complexity', 'N/A')}\n")
       
       # Groups
       markdown_content.append("## PR Groups\n")
       
       for i, group in enumerate(groups, 1):
           markdown_content.append(f"### PR {i}: {group.get('title', 'Untitled')}")
           markdown_content.append(f"\n**Branch**: `{group.get('branch', 'feature/update')}`")
           markdown_content.append(f"\n**Rationale**: {group.get('rationale', 'No rationale provided')}")
           markdown_content.append(f"\n**Files** ({len(group.get('files', []))}):")
           
           files = group.get('files', [])
           for file in files[:20]:  # Limit to first 20 files
               markdown_content.append(f"- `{file}`")
           
           if len(files) > 20:
               markdown_content.append(f"- ... and {len(files) - 20} more files")
           
           markdown_content.append("")  # Empty line between groups
       
       return {
           "format": "markdown",
           "content": "\n".join(markdown_content),
           "filename": f"pr_groups_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
       }
   
    def _export_as_csv(self, pr_grouping_strategy: Dict[str, Any]) -> Dict[str, Any]:
       """Export PR groups as CSV."""
       import csv
       import io
       
       output = io.StringIO()
       writer = csv.writer(output)
       
       # Header
       writer.writerow([
           "PR Number",
           "Title",
           "Branch Name",
           "File Count",
           "Files",
           "Rationale",
           "Directory Focus",
           "Feature Focus"
       ])
       
       # Data rows
       groups = pr_grouping_strategy.get('groups', [])
       for i, group in enumerate(groups, 1):
           files = group.get('files', [])
           writer.writerow([
               i,
               group.get('title', 'Untitled'),
               group.get('branch', f'feature/update-{i}'),
               len(files),
               ';'.join(files),  # Semicolon-separated file list
               group.get('rationale', ''),
               group.get('directory_focus', ''),
               group.get('feature_focus', '')
           ])
       
       return {
           "format": "csv",
           "content": output.getvalue(),
           "filename": f"pr_groups_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
       }