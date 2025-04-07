"""
Strategy selector tool for determining appropriate PR grouping strategies.
"""
from typing import List, Dict, Any, Optional
import json

from pydantic import Field, BaseModel, ValidationError

from shared.utils.logging_utils import get_logger
from .base_tools import BaseRepoTool

logger = get_logger(__name__)

class GroupingStrategySelectorInput(BaseModel):
    """Input schema for the StrategySelector tool."""
    repo_path: str = Field(..., description="Path to the git repository")
    repository_analysis: Dict[str, Any] = Field(..., description="Repository analysis data")

class GroupingStrategySelector(BaseRepoTool):
    """Tool for selecting appropriate grouping strategies based on repository characteristics."""

    name: str = "Strategy Selector"
    description: str = (
        "Selects the most appropriate grouping strategy type based on repository analysis and metrics. "
        "Outputs the chosen strategy type, rationale, and supporting data."
    )
    args_schema: type[BaseModel] = GroupingStrategySelectorInput

    def _run(self, repo_path: str, repository_analysis: Dict[str, Any], **kwargs) -> str:
        """
        Select appropriate grouping strategies based on repository analysis.

        Args:
            repo_path: Path to the git repository
            repository_analysis: Repository analysis data as dictionary
            **kwargs: Additional arguments (ignored)

        Returns:
            JSON string containing the selected strategy and recommendations
        """
        logger.info("Selecting grouping strategies")

        try:
            # Extract key information from the repository_analysis dictionary
            total_files = repository_analysis.get("total_files_changed", 0)
            total_lines = repository_analysis.get("total_lines_changed", 0)
            directory_summaries = repository_analysis.get("directory_summaries", [])
            file_changes = repository_analysis.get("file_changes", [])
            
            # Calculate directory counts and distribution
            directory_counts = {}
            for dir_summary in directory_summaries:
                dir_path = dir_summary.get("path", "")
                dir_file_count = dir_summary.get("file_count", 0)
                if dir_path:
                    directory_counts[dir_path] = dir_file_count
            
            # Calculate file types (extensions)
            # First try to use extensions_summary if available
            file_types = repository_analysis.get("extensions_summary", {})
            
            # If not available, extract from file changes
            if not file_types:
                file_types = {}
                for fc in file_changes:
                    extension = fc.get("extension", "")
                    if extension:
                        file_types[extension] = file_types.get(extension, 0) + 1
            
            # Calculate max files in a single directory
            max_dir_files = max(directory_counts.values()) if directory_counts else 0

            # Determine if most changes are concentrated in few directories
            directory_concentration = max_dir_files / total_files if total_files > 0 else 0

            # Determine if file counts are evenly distributed across multiple dirs
            significant_dirs = [count for count in directory_counts.values() if count >= max(2, total_files * 0.05)]
            is_distributed = len(significant_dirs) >= 3

            # Initialize an empty list of strategy recommendations
            recommendations = []

            # --- Strategy Recommendation Logic ---
            # Directory-based strategy
            if len(directory_counts) >= 2:
                conc_score = abs(directory_concentration - 0.5)
                confidence = min(0.9, 0.4 + conc_score + (len(directory_counts) * 0.05))
                
                concentration_level = "High" if directory_concentration > 0.7 else ("Low" if directory_concentration < 0.3 else "Moderate")
                
                recommendations.append({
                    "strategy_type": "directory_based",
                    "confidence": round(confidence, 2),
                    "rationale": (
                        f"Changes span {len(directory_counts)} directories. "
                        f"{concentration_level} concentration "
                        f"({directory_concentration:.2f}) suggests directory structure is a key factor."
                    ),
                    "estimated_pr_count": max(1, min(len(directory_counts), 7))
                })

            # Feature-based strategy
            if is_distributed and len(file_types) >= 3 and total_files > 5:
                confidence = 0.6 + (len(significant_dirs) * 0.03) + (len(file_types) * 0.02)
                recommendations.append({
                    "strategy_type": "feature_based",
                    "confidence": min(0.9, round(confidence, 2)),
                    "rationale": (
                        f"Changes are distributed across {len(significant_dirs)} significant directories and involve {len(file_types)} file types. "
                        "This pattern often indicates cross-cutting features."
                    ),
                    "estimated_pr_count": max(2, total_files // 6)
                })

            # Module-based strategy
            if len(file_types) >= 2 and directory_concentration < 0.6:
                confidence = 0.5 + (len(file_types) * 0.05) + (0.6 - directory_concentration) * 0.2
                
                # Get first 3 file types for message
                file_type_examples = list(file_types.keys())[:3]
                file_type_str = ', '.join(file_type_examples)
                if len(file_types) > 3:
                    file_type_str += "..."
                
                recommendations.append({
                    "strategy_type": "module_based",
                    "confidence": min(0.85, round(confidence, 2)),
                    "rationale": (
                        f"Changes include {len(file_types)} distinct file types ({file_type_str}) "
                        f"with relatively low directory concentration ({directory_concentration:.2f}). Grouping by module/type might be effective."
                    ),
                    "estimated_pr_count": max(2, min(len(file_types), 4))
                })

            # Size-balanced strategy
            if total_files > 15:
                confidence = 0.4 + (min(total_files, 100) / 250)
                recommendations.append({
                    "strategy_type": "size_balanced",
                    "confidence": min(0.8, round(confidence, 2)),
                    "rationale": (
                        f"With {total_files} changed files, splitting into size-balanced PRs ensures manageable reviews, "
                        "regardless of logical structure."
                    ),
                    "estimated_pr_count": max(2, total_files // 8)
                })

            # Default or Mixed strategy if no clear winner
            if not recommendations or max(rec.get("confidence", 0) for rec in recommendations) < 0.65:
                recommendations.append({
                    "strategy_type": "mixed",
                    "confidence": 0.7,
                    "rationale": (
                        "No single strategy stands out strongly based on current metrics. "
                        "A mixed approach, potentially combining directory and feature/module aspects, is recommended."
                    ),
                    "estimated_pr_count": max(2, min(total_files // 5, 7))
                })

            # Sort recommendations by confidence (highest first)
            recommendations.sort(key=lambda x: x.get("confidence", 0), reverse=True)

            # Select the highest confidence recommendation as the primary strategy
            primary_strategy_rec = recommendations[0] if recommendations else None
            primary_strategy_type = primary_strategy_rec.get("strategy_type", "mixed") if primary_strategy_rec else "mixed"

            # Create the explanation string
            if primary_strategy_rec:
                explanation = (
                    f"Selected '{primary_strategy_type}' as the primary strategy with "
                    f"{primary_strategy_rec.get('confidence', 0):.2f} confidence. "
                    f"Rationale: {primary_strategy_rec.get('rationale', '')}"
                )
            else:
                explanation = "No specific strategy strongly recommended; defaulting to MIXED."

            # Gather repository metrics used for the decision
            metrics_used = {
                "total_files_changed": total_files,
                "total_lines_changed": total_lines,
                "directory_count": len(directory_counts),
                "max_files_in_directory": max_dir_files,
                "directory_concentration": round(directory_concentration, 2),
                "file_type_count": len(file_types),
                "is_distributed": is_distributed,
            }

            # Create the result dictionary
            result = {
                "strategy_type": primary_strategy_type,
                "recommendations": recommendations,
                "repository_metrics": metrics_used,
                "explanation": explanation
            }

            # Return serialized JSON
            return json.dumps(result, indent=2)
            
        except Exception as e:
            error_msg = f"Error selecting grouping strategy: {str(e)}"
            logger.error(error_msg)
            
            # Return a serialized error response
            error_result = {
                "strategy_type": "mixed",  # Default to mixed as fallback
                "recommendations": [],
                "repository_metrics": {},
                "explanation": f"Error during strategy selection: {str(e)}",
                "error": error_msg
            }
            
            return json.dumps(error_result, indent=2)