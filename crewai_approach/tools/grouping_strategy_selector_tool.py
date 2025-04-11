# --- START OF FILE grouping_strategy_selector_tool.py ---
"""
Strategy selector tool for determining appropriate PR grouping strategies.
"""
from typing import List, Dict, Any, Optional, Type # Import Type
import json
from collections import defaultdict # Import defaultdict

from pydantic import Field, BaseModel, ValidationError

# Import Pydantic models and Enum
from models.agent_models import GroupingStrategyDecision, StrategyRecommendation, GroupingStrategyType
from shared.models.analysis_models import RepositoryAnalysis, FileChange, DirectorySummary
from shared.utils.logging_utils import get_logger
from .base_tool import BaseRepoTool

logger = get_logger(__name__)

class GroupingStrategySelectorSchema(BaseModel):
    """Input schema for the StrategySelector tool."""
    # repo_path is handled by BaseRepoTool
    repository_analysis_json: str = Field(..., description="JSON string serialization of the RepositoryAnalysis object.")
    # Add metrics/patterns if needed by logic
    # repository_metrics_json: Optional[str] = Field(None, description="Optional JSON string of RepositoryMetrics.")
    # pattern_analysis_json: Optional[str] = Field(None, description="Optional JSON string of PatternAnalysisResult.")


class GroupingStrategySelector(BaseRepoTool):
    """Tool for selecting appropriate grouping strategies based on repository characteristics."""

    name: str = "Strategy Selector"
    description: str = (
        "Selects the most appropriate grouping strategy type based on repository analysis JSON. " # Removed metrics mention unless explicitly passed
        "Outputs the chosen strategy type, rationale, and supporting data as a GroupingStrategyDecision JSON."
    )
    args_schema: Type[BaseModel] = GroupingStrategySelectorSchema # Correct type hint

    def _run(self, repository_analysis_json: str) -> str: # Keep kwargs for potential base class use
        """
        Select appropriate grouping strategies based on repository analysis JSON.

        Args:
            repository_analysis_json: JSON string of RepositoryAnalysis data.

        Returns:
            JSON string containing the selected strategy (GroupingStrategyDecision).
        """
        # Use repo_path from initialized git_ops if needed, e.g., for logging
        repo_path = self.git_ops.repo_path if hasattr(self, 'git_ops') and self.git_ops else "unknown_repo"
        logger.info(f"Selecting grouping strategy for repo: {repo_path}")

        try:
            # Deserialize the main input
            repository_analysis = RepositoryAnalysis.model_validate_json(repository_analysis_json)

            # *** Use attribute access for Pydantic objects ***
            total_files = repository_analysis.total_files_changed
            total_lines = repository_analysis.total_lines_changed
            directory_summaries: List[DirectorySummary] = repository_analysis.directory_summaries
            file_changes: List[FileChange] = repository_analysis.file_changes

            # --- Calculations needed for heuristics ---
            directory_counts = {ds.path: ds.file_count for ds in directory_summaries if ds.path}

            # Calculate file types (extensions) directly from file_changes
            file_types = defaultdict(int)
            for fc in file_changes:
                ext = fc.extension or "none"
                file_types[ext] += 1
            file_types = dict(file_types) # Convert back to dict

            max_dir_files = max(directory_counts.values()) if directory_counts else 0
            directory_concentration = max_dir_files / total_files if total_files > 0 else 0.0
            significant_dirs = [count for count in directory_counts.values() if count >= max(2, total_files * 0.05)]
            is_distributed = len(significant_dirs) >= 3
            # --- End Calculations ---

            recommendations_data: List[Dict[str, Any]] = []

            # --- Strategy Recommendation Logic ---
            # Directory-based strategy
            if len(directory_counts) >= 2:
                conc_score = abs(directory_concentration - 0.5)
                confidence = min(0.9, 0.4 + conc_score + (len(directory_counts) * 0.05))
                concentration_level = "High" if directory_concentration > 0.7 else ("Low" if directory_concentration < 0.3 else "Moderate")
                recommendations_data.append({
                    "strategy_type": GroupingStrategyType.DIRECTORY_BASED,
                    "confidence": round(confidence, 2),
                    "rationale": f"Changes span {len(directory_counts)} dirs. {concentration_level} concentration ({directory_concentration:.2f}) suggests directory structure is key.",
                    "estimated_pr_count": max(1, min(len(directory_counts), 7))
                })

            # Feature-based strategy
            if is_distributed and len(file_types) >= 3 and total_files > 5:
                confidence = 0.6 + (len(significant_dirs) * 0.03) + (len(file_types) * 0.02)
                recommendations_data.append({
                    "strategy_type": GroupingStrategyType.FEATURE_BASED,
                    "confidence": min(0.9, round(confidence, 2)),
                    "rationale": f"Changes across {len(significant_dirs)} dirs & {len(file_types)} file types often indicate cross-cutting features.",
                    "estimated_pr_count": max(2, total_files // 6)
                })

            # Module-based strategy
            if len(file_types) >= 2 and directory_concentration < 0.6:
                 confidence = 0.5 + (len(file_types) * 0.05) + (0.6 - directory_concentration) * 0.2
                 file_type_examples = list(file_types.keys())[:3]
                 file_type_str = ', '.join(file_type_examples) + ('...' if len(file_types) > 3 else '')
                 recommendations_data.append({
                     "strategy_type": GroupingStrategyType.MODULE_BASED,
                     "confidence": min(0.85, round(confidence, 2)),
                     "rationale": f"{len(file_types)} file types ({file_type_str}) with low concentration ({directory_concentration:.2f}). Grouping by type/module might work.",
                     "estimated_pr_count": max(2, min(len(file_types), 4))
                 })

            # Size-balanced strategy
            if total_files > 15:
                 confidence = 0.4 + (min(total_files, 100) / 250)
                 recommendations_data.append({
                     "strategy_type": GroupingStrategyType.SIZE_BALANCED,
                     "confidence": min(0.8, round(confidence, 2)),
                     "rationale": f"{total_files} changed files suggests splitting into balanced PRs ensures manageable reviews.",
                     "estimated_pr_count": max(2, total_files // 8)
                 })


            # --- Determine Primary Strategy ---
            if not recommendations_data or max((rec["confidence"] for rec in recommendations_data), default=0) < 0.65:
                primary_strategy_type = GroupingStrategyType.MIXED
                if not any(rec["strategy_type"] == GroupingStrategyType.MIXED for rec in recommendations_data):
                     recommendations_data.append({
                        "strategy_type": GroupingStrategyType.MIXED,
                        "confidence": 0.7,
                        "rationale": "No single strategy strongly indicated by metrics. A mixed approach is suggested.",
                        "estimated_pr_count": max(2, min(total_files // 5, 7))
                    })
            else:
                 recommendations_data.sort(key=lambda x: x["confidence"], reverse=True)
                 primary_strategy_type = recommendations_data[0]["strategy_type"]

            primary_strategy_rec_data = next((rec for rec in recommendations_data if rec["strategy_type"] == primary_strategy_type), recommendations_data[0] if recommendations_data else None)

            if primary_strategy_rec_data:
                explanation = (
                    f"Selected '{primary_strategy_type.value}' as the primary strategy with "
                    f"{primary_strategy_rec_data['confidence']:.2f} confidence. "
                    f"Rationale: {primary_strategy_rec_data['rationale']}"
                )
            else:
                explanation = "Could not determine a primary strategy; defaulting to MIXED."
                primary_strategy_type = GroupingStrategyType.MIXED

            metrics_used = {
                "total_files_changed": total_files,
                "total_lines_changed": total_lines, # Include if used in logic/explanation
                "directory_count": len(directory_counts),
                "max_files_in_directory": max_dir_files,
                "directory_concentration": round(directory_concentration, 2),
                "file_type_count": len(file_types),
                "is_distributed": is_distributed,
            }

            # Create the final result object using Pydantic model
            result = GroupingStrategyDecision(
                strategy_type=primary_strategy_type,
                recommendations=[StrategyRecommendation(**rec) for rec in recommendations_data],
                repository_metrics=metrics_used,
                explanation=explanation
            )

            # *** Return serialized JSON ***
            logger.info(f"Strategy selection complete for {repo_path}. Primary: {primary_strategy_type.value}")
            return result.model_dump_json(indent=2)

        # Keep error handling similar, ensure it returns valid JSON string
        except json.JSONDecodeError as je:
            error_msg = f"Failed to decode input repository_analysis_json: {str(je)}"
            logger.error(error_msg, exc_info=True)
            error_result = GroupingStrategyDecision(strategy_type=GroupingStrategyType.MIXED, explanation=error_msg, error=error_msg)
            return error_result.model_dump_json(indent=2)
        except ValidationError as ve:
             # This could happen if the input JSON doesn't match RepositoryAnalysis model
             error_msg = f"Input validation error during strategy selection: {str(ve)}"
             logger.error(error_msg, exc_info=True)
             error_result = GroupingStrategyDecision(strategy_type=GroupingStrategyType.MIXED, explanation=error_msg, error=error_msg)
             return error_result.model_dump_json(indent=2)
        except Exception as e:
            # Catch other errors, including potential AttributeErrors if models change
            error_msg = f"Error selecting grouping strategy: {str(e)}"
            logger.error(error_msg, exc_info=True)
            error_result = GroupingStrategyDecision(strategy_type=GroupingStrategyType.MIXED, explanation=error_msg, error=error_msg)
            return error_result.model_dump_json(indent=2)

# --- END OF FILE grouping_strategy_selector_tool.py ---