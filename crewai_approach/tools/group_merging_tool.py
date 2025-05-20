# tools/group_merging_tool.py
import json
import re
from typing import Type, List, Dict, Set, Optional, Any

from pydantic import BaseModel, Field, ValidationError

from .base_tool import BaseRepoTool
from crewai_approach.models.agent_models import PRGroupingStrategy, PRGroup, GroupingStrategyType
from shared.utils.logging_utils import get_logger
from shared.models.analysis_models import RepositoryAnalysis, FileChange

logger = get_logger(__name__)

# --- REVERTED SCHEMA ---
class GroupMergingToolSchema(BaseModel):
    """Input schema for GroupMergingTool using primitive types."""
    # Expect a JSON *string* representing a list of batch result strings/objects
    batch_grouping_results_json: str = Field(..., description="REQUIRED: A JSON array *string*, where each element is the JSON serialization of a PRGroupingStrategy object from one batch.")
    original_repository_analysis_json: str = Field(..., description="REQUIRED: JSON string of the original, full RepositoryAnalysis object.")

class GroupMergingTool(BaseRepoTool):
    name: str = "Group Merging Tool"
    description: str = "Merges PR grouping results (provided as a JSON array string) from multiple batches into a single, coherent PRGroupingStrategy JSON string."
    args_schema: Type[BaseModel] = GroupMergingToolSchema

    # --- Helper to clean potential markdown/whitespace ---
    def _clean_json_string(self, json_string: Optional[str]) -> Optional[str]:
         if not json_string or not isinstance(json_string, str):
             return None
         try:
             cleaned = re.sub(r'^```json\s*', '', json_string.strip(), flags=re.MULTILINE)
             cleaned = re.sub(r'```\s*$', '', cleaned, flags=re.MULTILINE).strip()
             # Only return if it looks like JSON, let the main parser handle validation
             if cleaned.startswith('{') or cleaned.startswith('['):
                  return cleaned
             else: return None
         except: return None

    def _extract_file_paths(self, repo_analysis_json: Optional[str]) -> Set[str]:
         if not repo_analysis_json: return set()
         try:
             repo_analysis = RepositoryAnalysis.model_validate_json(repo_analysis_json)
             if repo_analysis.file_changes:
                 return {fc.path for fc in repo_analysis.file_changes if fc.path}
         except (ValidationError, json.JSONDecodeError) as e:
             logger.warning(f"Could not extract file paths from repo analysis JSON: {e}")
         return set()

    def _run(
        self,
        batch_grouping_results_json: str, # Expects JSON string of list
        original_repository_analysis_json: str,
    ) -> str:
        """Merges batch grouping results (provided as a JSON array string)."""
        logger.info(f"GroupMergingTool received batch_grouping_results_json: {batch_grouping_results_json[:150]}...")
        logger.info(f"GroupMergingTool received original_repository_analysis_json: {original_repository_analysis_json[:100]}...")

        final_pr_groups: List[PRGroup] = []
        all_grouped_files: Set[str] = set()
        overall_explanation = "Merged groups from batches.\n"
        strategy_type_value = "mixed" # Default
        batch_results_dicts: List[Dict[str, Any]] = [] # To store parsed dicts

        try:
            # --- Clean and Parse Inputs ---
            cleaned_batch_results_str = self._clean_json_string(batch_grouping_results_json)
            cleaned_repo_analysis_str = self._clean_json_string(original_repository_analysis_json)

            if not cleaned_batch_results_str:
                 raise ValueError("Invalid or empty batch_grouping_results_json provided.")
            if not cleaned_repo_analysis_str:
                 raise ValueError("Invalid or empty original_repository_analysis_json provided.")

            # Parse the batch results string into a list of dictionaries
            try:
                batch_results_parsed = json.loads(cleaned_batch_results_str)
                if not isinstance(batch_results_parsed, list):
                     raise TypeError("Parsed batch_grouping_results_json is not a list.")
                # Further check if list items are dictionaries (optional, handled below)
                batch_results_dicts = [item for item in batch_results_parsed if isinstance(item, dict)]
                logger.info(f"Successfully parsed {len(batch_results_dicts)} batch result dictionaries from JSON string.")
            except (json.JSONDecodeError, TypeError) as e:
                raise ValueError(f"Failed to parse batch_grouping_results_json string into a list: {e}") from e

            if not batch_results_dicts:
                 logger.warning("Parsed batch results list is empty or contains no dictionaries.")
                 empty_strategy = PRGroupingStrategy(strategy_type=GroupingStrategyType.MIXED, groups=[], explanation="No valid batch results provided.")
                 return empty_strategy.model_dump_json(indent=2)

            original_file_paths = self._extract_file_paths(cleaned_repo_analysis_str)
            if not original_file_paths:
                 logger.warning("Could not extract any file paths from original_repository_analysis_json.")

            # --- Merging Logic (operates on dictionaries) ---
            merged_groups_dicts: List[Dict[str, Any]] = []

            for batch_dict in batch_results_dicts:
                if not isinstance(batch_dict, dict) or "groups" not in batch_dict:
                    logger.warning(f"Skipping invalid batch result dictionary: {str(batch_dict)[:100]}...")
                    continue

                if strategy_type_value == "mixed": strategy_type_value = batch_dict.get("strategy_type", "mixed")
                groups_in_batch = batch_dict.get("groups", [])
                if isinstance(groups_in_batch, list): merged_groups_dicts.extend(g for g in groups_in_batch if isinstance(g, dict))
                batch_explanation = batch_dict.get("explanation", "No explanation")
                overall_explanation += f"\nBatch ({batch_dict.get('strategy_type', 'N/A')}): {batch_explanation}"
                for group_dict in groups_in_batch:
                    if isinstance(group_dict, dict):
                        files = group_dict.get("files", [])
                        if isinstance(files, list): all_grouped_files.update(f for f in files if isinstance(f, str))

            logger.info(f"Initially merged {len(merged_groups_dicts)} group dictionaries covering {len(all_grouped_files)} files.")

            # --- Deduplication ---
            final_groups_dicts: List[Dict[str, Any]] = []
            seen_files: Set[str] = set()
            for group_dict in merged_groups_dicts:
                files = group_dict.get("files", [])
                if files and isinstance(files, list):
                    unique_files_in_group = [f for f in files if isinstance(f, str) and f not in seen_files]
                    if unique_files_in_group:
                        new_group_dict = dict(group_dict); new_group_dict["files"] = unique_files_in_group
                        final_groups_dicts.append(new_group_dict); seen_files.update(unique_files_in_group)

            unmerged_files = list(original_file_paths - seen_files)
            notes = f"Merged {len(batch_results_dicts)} batches. Found {len(unmerged_files)} unmerged files."
            if unmerged_files: logger.warning(f"{notes}")

            # --- Convert to Pydantic Models ---
            # ... (Keep the conversion logic from PRGroup dictionaries to Pydantic objects as before) ...
            try: strategy_type_enum = GroupingStrategyType(strategy_type_value)
            except: strategy_type_enum = GroupingStrategyType.MIXED
            for group_dict in final_groups_dicts:
                 try:
                     files = [f for f in group_dict.get("files", []) if isinstance(f, str)]
                     if not files: continue
                     if "title" not in group_dict: group_dict["title"] = "Untitled Merged Group"
                     pr_group = PRGroup(title=str(group_dict.get("title")), files=files, rationale=str(group_dict.get("rationale", "Merged.")), estimated_size=int(group_dict.get("estimated_size", len(files))), directory_focus=group_dict.get("directory_focus"), feature_focus=group_dict.get("feature_focus"), suggested_branch_name=group_dict.get("suggested_branch_name"), suggested_pr_description=group_dict.get("suggested_pr_description"))
                     final_pr_groups.append(pr_group)
                 except (ValidationError, TypeError) as e: logger.warning(f"Skipping group due to Pydantic error: {e}. Group: {str(group_dict)[:100]}...")

            # --- Create final strategy ---
            merged_strategy = PRGroupingStrategy(strategy_type=strategy_type_enum, groups=final_pr_groups, explanation=overall_explanation.strip() + "\n" + notes, estimated_review_complexity=self._estimate_final_complexity(final_pr_groups), ungrouped_files=unmerged_files)
            result_json = merged_strategy.model_dump_json(indent=2); json.loads(result_json) # Validate output
            logger.info(f"Successfully created merged strategy JSON with {len(final_pr_groups)} groups.")
            return result_json

        except Exception as e:
             error_msg = f"Error in GroupMergingTool: {e}"
             logger.error(error_msg, exc_info=True)
             error_strategy = PRGroupingStrategy(strategy_type=GroupingStrategyType.MIXED, groups=[], explanation=f"Merging failed: {e}", ungrouped_files=[])
             return error_strategy.model_dump_json(indent=2)

    def _estimate_final_complexity(self, groups: List[PRGroup]) -> float:
         if not groups: return 1.0
         group_count = len(groups); total_files = sum(len(g.files) for g in groups if g.files); max_files = max((len(g.files) for g in groups if g.files), default=0)
         complexity = 1.0 + (group_count * 0.6) + (total_files * 0.08) + (max_files * 0.04)
         return min(10.0, max(1.0, round(complexity, 1)))