"""
Enhanced PR Generator with intelligent file grouping strategies.
"""
import logging
import os
from typing import Dict, List, Set, Any, Optional
from collections import defaultdict

from shared.config.llm_config import LLMConfig, create_llm_config
from llm_guided_approach.llm_service import LLMService
from llm_guided_approach.generators.description_generator import DescriptionGenerator
from llm_guided_approach.generators.file_grouper_with_embeddings import FileGrouperWithEmbeddings
from shared.utils.logging_utils import log_operation
from shared.models.pr_models import (
    FileChange, PRSuggestion, PullRequestGroup
)
from shared.git_operations import get_changed_files

logger = logging.getLogger(__name__)

class PRGenerator:
    """
    PR Generator with intelligent file grouping strategies.
    
    Features:
    1. Directory-based initial grouping
    2. Token-aware chunking
    3. LLM-based semantic grouping
    4. Smart handling of outlier files
    """
    
    def __init__(
        self,
        repo_path: str,
        max_files_per_pr: int = 20,
        max_tokens_per_request: int = 4000,
        llm_provider: str = "openai",
        llm_model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        verbose: bool = False
    ):
        """Initialize the Enhanced PR Generator."""
        self.repo_path = repo_path
        self.max_files_per_pr = max_files_per_pr
        self.max_tokens_per_request = max_tokens_per_request
        self.verbose = verbose
        
        # Create LLM configuration and service
        self.llm_config = create_llm_config(
            llm_provider=llm_provider,
            llm_model=llm_model,
            api_key=api_key,
            base_url=base_url,
            temperature=0.2
        )
        
        self.llm_service = LLMService(self.llm_config)
        
        # Initialize description generator
        self.description_generator = DescriptionGenerator(self.llm_service, verbose)
        
    @log_operation("Creating PRs with smart grouping")
    def create_prs(self) -> Dict[str, Any]:
        """Create PR suggestions with intelligent grouping."""
        try:
            # Get all changed files
            changes = get_changed_files(self.repo_path)
            all_files = {change.file_path for change in changes}
            logger.info(f"Found {len(all_files)} changed files")

            # 1. Initial semantic grouping with embeddings
            embedding_grouper = FileGrouperWithEmbeddings(self.llm_service, self.verbose)
            semantic_groups = embedding_grouper.group_files(changes)
            logger.info(f"Initial semantic groups: {len(semantic_groups)}")

            # 2. Merge small groups by directory similarity
            merged_groups = self._merge_small_groups(semantic_groups)
            logger.info(f"Merged groups: {len(merged_groups)}")

            # 3. Process groups with token-aware chunking
            all_processed_groups = []
            for group in merged_groups:
                group_changes = [c for c in changes if c.file_path in group["files"]]
                
                # Split into token-aware chunks
                chunks = self._chunk_by_token_budget(group_changes)
                
                for i, chunk in enumerate(chunks):
                    chunk_name = f"{group['title']} Part {i+1}" if len(chunks) > 1 else group['title']
                    processed = self._process_group(chunk, chunk_name)
                    all_processed_groups.extend(processed)

            # 4. Handle remaining outliers
            grouped_files = set()
            for group in all_processed_groups:
                grouped_files.update(group.get("files", []))
                
            outliers = [c for c in changes if c.file_path not in grouped_files]
            if outliers:
                logger.info(f"Processing {len(outliers)} outlier files")
                outlier_groups = self._handle_outliers(outliers)
                all_processed_groups.extend(outlier_groups)

            # 5. Final merge pass for any small groups
            final_groups = self._merge_small_groups(all_processed_groups)
            logger.info(f"Final grouped PRs: {len(final_groups)}")

            # Create PR suggestions
            pr_suggestions, validation_result = self._create_pr_suggestions(final_groups)
            
            return PRSuggestion(
                pr_suggestions=pr_suggestions,
                total_groups=len(pr_suggestions),
                message=f"Created {len(pr_suggestions)} pull request suggestions",
                validation_result=validation_result
            ).model_dump()
            
        except Exception as e:
            logger.exception(f"PR generation failed: {e}")
            return {"error": str(e), "pr_suggestions": []}

    def _merge_small_groups(self, groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge groups with fewer than 3 files based on directory similarity."""
        merged = []
        current_group = None
        
        # Sort groups by primary directory
        sorted_groups = sorted(groups, key=lambda g: os.path.dirname(g["files"][0]) if g["files"] else "")

        for group in sorted_groups:
            if not current_group:
                current_group = group
                continue
                
            # Get common parent directory
            current_dir = os.path.dirname(current_group["files"][0]) if current_group["files"] else ""
            new_dir = os.path.dirname(group["files"][0]) if group["files"] else ""
            
            # Find common parent directory
            common_parent = os.path.commonpath([current_dir, new_dir])
            
            # Merge conditions
            if (common_parent and 
                len(current_group["files"]) + len(group["files"]) <= self.max_files_per_pr and
                len(current_group["files"]) < 3):
                current_group["files"].extend(group["files"])
                current_group["title"] = f"{common_parent} updates"
                current_group["reasoning"] = f"Combined changes in {common_parent} directory"
            else:
                merged.append(current_group)
                current_group = group
                
        if current_group:
            merged.append(current_group)
            
        return merged

    def _chunk_by_token_budget(self, files: List[FileChange]) -> List[List[FileChange]]:
        """Improved token-aware chunking with large files first."""
        chunks = []
        
        # Sort files by diff size descending
        sorted_files = sorted(files, key=lambda f: -len(f.diff))
        
        current_chunk = []
        current_tokens = 0
        
        for file in sorted_files:
            file_tokens = self._estimate_tokens(file)
            
            # Start new chunk if exceeds 90% of budget
            if current_tokens + file_tokens > self.max_tokens_per_request * 0.9:
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0
                
            current_chunk.append(file)
            current_tokens += file_tokens
            
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks

    def _estimate_tokens(self, file: FileChange) -> int:
        """Improved token estimation with diff content."""
        base_tokens = 100  # Metadata tokens
        if file.diff:
            return base_tokens + (len(file.diff) // 3)  # ~3 chars per token
        return base_tokens
    
    def _group_by_directory(self, changes: List[FileChange]) -> Dict[str, List[FileChange]]:
        """Group files by directory."""
        directory_groups = defaultdict(list)
        for change in changes:
            directory_groups[change.directory].append(change)
        return directory_groups
    
    def _chunk_by_token_budget(self, files: List[FileChange]) -> List[List[FileChange]]:
        """Improved token-aware chunking"""
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for file in sorted(files, key=lambda f: -len(f.diff)):
            file_tokens = self._estimate_tokens(file)
            
            if current_tokens + file_tokens > self.max_tokens_per_request * 0.9:
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0
                
            current_chunk.append(file)
            current_tokens += file_tokens
            
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks

    def _estimate_tokens(self, file: FileChange) -> int:
        """More accurate token estimation"""
        if file.diff:
            return len(file.diff) // 3  # Approx 3 chars per token
        return 100  # Base tokens for metadata
    
    def _process_group(self, files: List[FileChange], group_name: str) -> List[Dict[str, Any]]:
        """Process a group of files with the LLM."""
        # Create a prompt for the LLM
        grouping_prompt = self.llm_service.create_grouping_prompt(files)
        
        # Get grouping suggestions from the LLM
        result = self.llm_service.analyze_changes(grouping_prompt)
        
        # Extract groups from the result
        if "groups" in result and result["groups"]:
            return result["groups"]
        
        # Fallback if LLM didn't provide valid groups
        return [{
            "title": f"Update {group_name}",
            "files": [f.file_path for f in files],
            "reasoning": f"Changes to files in the {group_name} component",
            "branch_name": f"update-{group_name.replace('/', '-').lower()}"
        }]
    
    def _handle_outliers(self, outliers: List[FileChange]) -> List[Dict[str, Any]]:
        """Handle outlier files (single files in directories)."""
        # If there are only a few outliers, group them together
        if len(outliers) <= self.max_files_per_pr:
            return [{
                "title": "Miscellaneous Updates",
                "files": [f.file_path for f in outliers],
                "reasoning": "Collection of smaller updates across various components",
                "branch_name": "misc-updates"
            }]
        
        # For many outliers, try to group them by similarity
        # (simplified approach - in practice, consider using embeddings)
        
        # Group by file extension/type
        by_extension = defaultdict(list)
        for file in outliers:
            ext = file.extension or "other"
            by_extension[ext].append(file)
        
        groups = []
        for ext, ext_files in by_extension.items():
            # If this group is too large, we could chunk it further
            groups.append({
                "title": f"Update {ext.upper()} Files",
                "files": [f.file_path for f in ext_files],
                "reasoning": f"Updates to {ext.upper()} files across the project",
                "branch_name": f"update-{ext}-files"
            })
        
        return groups
    
    def _generate_branch_name(self, title: str) -> str:
        """Generate a simple git branch name from a PR title."""
        # Convert to lowercase, replace spaces with hyphens
        branch = title.lower().replace(" ", "-")
        # Keep only alphanumeric characters and hyphens
        branch = ''.join(c for c in branch if c.isalnum() or c == '-')
        # Truncate
        return branch[:50]
    
    def _create_pr_suggestions(self, groups: List[Dict[str, Any]]) -> List[PullRequestGroup]:
        """Create PullRequestGroup objects with inline validation."""
        pr_suggestions = []
        seen_titles = set()
        validation_issues = []
        
        # First pass: filter out empty groups
        valid_groups = []
        for group in groups:
            if not group.get("files", []):
                logger.warning(f"Skipping empty group: {group.get('title', 'Untitled')}")
                validation_issues.append(f"Empty group: {group.get('title', 'Untitled')}")
                continue
            valid_groups.append(group)
        
        for group in valid_groups:
            # Ensure unique title
            title = group.get("title", "Untitled PR")
            counter = 1
            original_title = title
            
            while title in seen_titles:
                title = f"{original_title} ({counter})"
                counter += 1
            
            seen_titles.add(title)
            
            # Generate branch name if missing
            suggested_branch = group.get("suggested_branch", "")
            if not suggested_branch:
                suggested_branch = self._generate_branch_name(title)
            
            # Create PullRequestGroup
            pr_group = PullRequestGroup(
                title=title,
                description=group.get("description", group.get("reasoning", "")),
                files=group.get("files", []),
                rationale=group.get("reasoning", group.get("rationale", "")),
                suggested_branch=suggested_branch
            )
            
            pr_suggestions.append(pr_group)
        
        # Create validation result
        validation_result = {
            "valid": len(validation_issues) == 0,
            "issues": validation_issues,
            "stats": {
                "total_groups": len(pr_suggestions),
                "filtered_groups": len(groups) - len(valid_groups)
            }
        }
        
        # Return PRSuggestion with embedded validation
        return pr_suggestions, validation_result