"""
Strategy selector tool for determining appropriate PR grouping strategies.
"""
from typing import List, Dict, Any, Optional, Type
import json
from collections import defaultdict

from pydantic import Field, BaseModel, ValidationError

# Import Pydantic models and Enum
from models.agent_models import GroupingStrategyDecision, StrategyRecommendation, GroupingStrategyType
from shared.utils.logging_utils import get_logger
from .base_tool import BaseRepoTool

logger = get_logger(__name__)

class GroupingStrategySelectorSchema(BaseModel):
    """Input schema for the StrategySelector tool using primitive types."""
    repository_analysis_json: str = Field(..., description="JSON string serialization of the RepositoryAnalysis object.")
    # Optional metrics/patterns if needed
    repository_metrics_json: Optional[str] = Field(None, description="Optional JSON string of RepositoryMetrics.")
    pattern_analysis_json: Optional[str] = Field(None, description="Optional JSON string of PatternAnalysisResult.")


class GroupingStrategySelector(BaseRepoTool):
    """Tool for selecting appropriate grouping strategies based on repository characteristics."""

    name: str = "Strategy Selector"
    description: str = (
        "Selects the most appropriate grouping strategy type based on repository analysis JSON. "
        "Outputs the chosen strategy type, rationale, and supporting data as a GroupingStrategyDecision JSON."
    )
    args_schema: Type[BaseModel] = GroupingStrategySelectorSchema

    def _run(
        self, 
        repository_analysis_json: str,
        repository_metrics_json: Optional[str] = None,
        pattern_analysis_json: Optional[str] = None
    ) -> str:
        """
        Select appropriate grouping strategies based on repository analysis JSON.

        Args:
            repository_analysis_json: JSON string of RepositoryAnalysis data.
            repository_metrics_json: Optional JSON string of RepositoryMetrics data.
            pattern_analysis_json: Optional JSON string of PatternAnalysisResult data.

        Returns:
            JSON string containing the selected strategy (GroupingStrategyDecision).
        """
        # Echo received inputs for debugging
        logger.info(f"GroupingStrategySelector received repository_analysis_json: {repository_analysis_json[:100]}...")
        if repository_metrics_json:
            logger.info(f"GroupingStrategySelector received repository_metrics_json: {repository_metrics_json[:100]}...")
        if pattern_analysis_json:
            logger.info(f"GroupingStrategySelector received pattern_analysis_json: {pattern_analysis_json[:100]}...")
        
        # Use repo_path from initialized git_ops if needed, e.g., for logging
        repo_path = self.git_ops.repo_path if hasattr(self, 'git_ops') and self.git_ops else "unknown_repo"
        logger.info(f"Selecting grouping strategy for repo: {repo_path}")

        try:
            # Validate input JSON
            if not self._validate_json_string(repository_analysis_json):
                raise ValueError("Invalid repository_analysis_json provided")
            
            # If repository_metrics_json is provided, use it instead of calculating metrics
            if repository_metrics_json and self._validate_json_string(repository_metrics_json):
                metrics_data = self._safe_deserialize(repository_metrics_json)
                # Extract metrics directly
                directory_metrics = metrics_data.get("directory_metrics", {})
                file_type_metrics = metrics_data.get("file_type_metrics", {})
                directory_concentration = directory_metrics.get("directory_concentration", 0.0)
                total_files = metrics_data.get("total_files_changed", 0)
                total_lines = metrics_data.get("total_lines_changed", 0)
                file_types = file_type_metrics.get("file_type_distribution", {})
            else:
                # Calculate metrics from repository analysis
                repo_info = self._extract_repository_info(repository_analysis_json)
                total_files = repo_info.get("total_files_changed", 0)
                total_lines = repo_info.get("total_lines_changed", 0)
                
                # Extract directory summaries
                directory_summaries = self._extract_directory_summaries(repository_analysis_json)
                
                # Extract file metadata to get file types
                file_metadata = self._extract_file_metadata(repository_analysis_json)
                
                # Calculate directory counts
                directory_counts = {}
                for ds in directory_summaries:
                    path = ds.get("path", "")
                    if path:
                        directory_counts[path] = ds.get("file_count", 0)
                
                # Calculate file types
                file_types = defaultdict(int)
                for fm in file_metadata:
                    ext = fm.get("extension", "") or "none"
                    file_types[ext] += 1
                file_types = dict(file_types)
                
                # Calculate directory concentration
                max_dir_files = max(directory_counts.values()) if directory_counts else 0
                directory_concentration = max_dir_files / total_files if total_files > 0 else 0.0
            
            # Calculate significant directories (dirs with at least 5% of changed files)
            significant_dirs = []
            if "directory_counts" in locals():
                significant_dirs = [count for count in directory_counts.values() 
                                  if count >= max(2, total_files * 0.05)]
            else:
                # Get from metrics if available
                directory_metrics_dict = metrics_data.get("directory_metrics", {}) if "metrics_data" in locals() else {}
                if "directory_distribution" in directory_metrics_dict:
                    for dir_path, percentage in directory_metrics_dict.get("directory_distribution", {}).items():
                        file_count = int((percentage / 100) * total_files)
                        if file_count >= max(2, total_files * 0.05):
                            significant_dirs.append(file_count)
            
            is_distributed = len(significant_dirs) >= 3

            recommendations_data: List[Dict[str, Any]] = []

            # --- Strategy Recommendation Logic ---
            # Directory-based strategy
            directory_count = len(directory_counts) if "directory_counts" in locals() else 0
            if directory_count >= 2:
                conc_score = abs(directory_concentration - 0.5)
                confidence = min(0.9, 0.4 + conc_score + (directory_count * 0.05))
                concentration_level = "High" if directory_concentration > 0.7 else ("Low" if directory_concentration < 0.3 else "Moderate")
                recommendations_data.append({
                    "strategy_type": GroupingStrategyType.DIRECTORY_BASED,
                    "confidence": round(confidence, 2),
                    "rationale": f"Changes span {directory_count} dirs. {concentration_level} concentration ({directory_concentration:.2f}) suggests directory structure is key.",
                    "estimated_pr_count": max(1, min(directory_count, 7))
                })

            # Feature-based strategy
            file_type_count = len(file_types)
            if is_distributed and file_type_count >= 3 and total_files > 5:
                confidence = 0.6 + (len(significant_dirs) * 0.03) + (file_type_count * 0.02)
                recommendations_data.append({
                    "strategy_type": GroupingStrategyType.FEATURE_BASED,
                    "confidence": min(0.9, round(confidence, 2)),
                    "rationale": f"Changes across {len(significant_dirs)} dirs & {file_type_count} file types often indicate cross-cutting features.",
                    "estimated_pr_count": max(2, total_files // 6)
                })

            # Module-based strategy
            if file_type_count >= 2 and directory_concentration < 0.6:
                 confidence = 0.5 + (file_type_count * 0.05) + (0.6 - directory_concentration) * 0.2
                 file_type_examples = list(file_types.keys())[:3]
                 file_type_str = ', '.join(file_type_examples) + ('...' if file_type_count > 3 else '')
                 recommendations_data.append({
                     "strategy_type": GroupingStrategyType.MODULE_BASED,
                     "confidence": min(0.85, round(confidence, 2)),
                     "rationale": f"{file_type_count} file types ({file_type_str}) with low concentration ({directory_concentration:.2f}). Grouping by type/module might work.",
                     "estimated_pr_count": max(2, min(file_type_count, 4))
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
                "total_lines_changed": total_lines,
                "directory_count": directory_count if "directory_count" in locals() else 0,
                "max_files_in_directory": max(directory_counts.values()) if "directory_counts" in locals() and directory_counts else 0,
                "directory_concentration": round(directory_concentration, 2),
                "file_type_count": file_type_count,
                "is_distributed": is_distributed,
            }

            # Create the final result object using Pydantic model
            result = GroupingStrategyDecision(
                strategy_type=primary_strategy_type,
                recommendations=[StrategyRecommendation(**rec) for rec in recommendations_data],
                repository_metrics=metrics_used,
                explanation=explanation
            )

            # Return serialized JSON
            logger.info(f"Strategy selection complete for {repo_path}. Primary: {primary_strategy_type.value}")
            return result.model_dump_json(indent=2)

        # Keep error handling similar, ensure it returns valid JSON string
        except json.JSONDecodeError as je:
            error_msg = f"Failed to decode input repository_analysis_json: {str(je)}"
            logger.error(error_msg, exc_info=True)
            error_result = GroupingStrategyDecision(
                strategy_type=GroupingStrategyType.MIXED, 
                recommendations=[],
                repository_metrics={},
                explanation=error_msg
            )
            return error_result.model_dump_json(indent=2)
        except ValidationError as ve:
             # This could happen if the input JSON doesn't match RepositoryAnalysis model
             error_msg = f"Input validation error during strategy selection: {str(ve)}"
             logger.error(error_msg, exc_info=True)
             error_result = GroupingStrategyDecision(
                 strategy_type=GroupingStrategyType.MIXED, 
                 recommendations=[],
                 repository_metrics={},
                 explanation=error_msg
             )
             return error_result.model_dump_json(indent=2)
        except Exception as e:
            # Catch other errors, including potential AttributeErrors if models change
            error_msg = f"Error selecting grouping strategy: {str(e)}"
            logger.error(error_msg, exc_info=True)
            error_result = GroupingStrategyDecision(
                strategy_type=GroupingStrategyType.MIXED, 
                recommendations=[],
                repository_metrics={},
                explanation=error_msg
            )
            return error_result.model_dump_json(indent=2)