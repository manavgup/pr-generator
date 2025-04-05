"""
Strategy selector tool for determining appropriate PR grouping strategies.
"""
from typing import List, Dict, Any, Optional
from pydantic import Field, BaseModel

from shared.utils.logging_utils import get_logger
# Import the specific models needed
from models.agent_models import GroupingStrategyType, StrategyRecommendation, GroupingStrategyDecision # Import new output model
# Import base tool and input models
from .base_tools import BaseRepoTool # Assuming BasePROrganizationTool inherits from BaseRepoTool or similar structure
from models.agent_models import RepositoryMetrics # If using RepositoryMetrics directly
# Import RepositoryAnalysis for type hinting
from shared.models.analysis_models import RepositoryAnalysis

logger = get_logger(__name__)


class GroupingStrategySelectorInput(BaseModel): # Use BaseModel if RepoToolInput isn't strictly needed or defined elsewhere
    """Input schema for the StrategySelector tool."""
    repo_path: str = Field(..., description="Path to the git repository")
    # Use RepositoryAnalysis directly for type safety
    repository_analysis: RepositoryAnalysis = Field(..., description="Repository analysis data")
    # Optionally include metrics if calculated separately
    # repository_metrics: Optional[RepositoryMetrics] = Field(None, description="Pre-calculated repository metrics")

# Update the generic type hint for the output model
class GroupingStrategySelector(BaseRepoTool): # Assuming BaseRepoTool is the correct base
    """Tool for selecting appropriate grouping strategies based on repository characteristics."""

    name: str = "Strategy Selector"
    description: str = (
        "Selects the most appropriate grouping strategy type based on repository analysis and metrics. "
        "Outputs the chosen strategy type, rationale, and supporting data."
    )
    args_schema: type[BaseModel] = GroupingStrategySelectorInput

    # Update the _run signature and return type hint
    def _run(self, repo_path: str, repository_analysis: RepositoryAnalysis, **kwargs) -> GroupingStrategyDecision:
        """
        Select appropriate grouping strategies based on repository analysis.

        Args:
            repo_path: Path to the git repository
            repository_analysis: RepositoryAnalysis object containing repository analysis data

        Returns:
            StrategySelection object with the selected strategy type, recommendations, and rationale.
        """
        logger.info("Selecting grouping strategies")

        # Extract key information directly from the RepositoryAnalysis object
        total_files = repository_analysis.total_files_changed
        total_lines = repository_analysis.total_lines_changed # Assuming this exists or can be calculated
        directory_summaries = repository_analysis.directory_summaries
        file_changes = repository_analysis.file_changes

        # Use computed properties or recalculate if needed
        directory_counts = {
            dir_summary.path: dir_summary.file_count
            for dir_summary in directory_summaries
        }
        file_types = repository_analysis.extensions_summary # Use computed property


        # Calculate max files in a single directory
        max_dir_files = max(directory_counts.values()) if directory_counts else 0

        # Determine if most changes are concentrated in few directories
        directory_concentration = max_dir_files / total_files if total_files > 0 else 0

        # Determine if file counts are evenly distributed across multiple dirs
        # Refined logic: check if std dev is low relative to mean, or if multiple dirs have significant counts
        significant_dirs = [count for count in directory_counts.values() if count >= max(2, total_files * 0.05)] # Dirs with >5% of files or at least 2
        is_distributed = len(significant_dirs) >= 3


        # Initialize an empty list of strategy recommendations
        recommendations: List[StrategyRecommendation] = []

        # --- Strategy Recommendation Logic (Keep as is) ---

        # Directory-based strategy
        # Condition: More than one directory affected, changes show some concentration or clear dir structure
        if len(directory_counts) >= 2:
             # Confidence higher if concentration is high OR low (clear structure vs many distinct areas)
             conc_score = abs(directory_concentration - 0.5) # Score closer to 0.5 is less confident
             confidence = min(0.9, 0.4 + conc_score + (len(directory_counts) * 0.05)) # Base + concentration + num_dirs bonus
             recommendations.append(StrategyRecommendation(
                 strategy_type=GroupingStrategyType.DIRECTORY_BASED,
                 confidence=round(confidence, 2),
                 rationale=(
                     f"Changes span {len(directory_counts)} directories. "
                     f"{'High' if directory_concentration > 0.7 else ('Low' if directory_concentration < 0.3 else 'Moderate')} concentration "
                     f"({directory_concentration:.2f}) suggests directory structure is a key factor."
                 ),
                 estimated_pr_count=max(1, min(len(directory_counts), 7)) # Cap estimated PRs
             ))

        # Feature-based strategy
        # Condition: Changes are distributed, multiple file types involved, suggesting cross-cutting concerns
        if is_distributed and len(file_types) >= 3 and total_files > 5:
             confidence = 0.6 + (len(significant_dirs) * 0.03) + (len(file_types) * 0.02) # Base + bonus for distribution/types
             recommendations.append(StrategyRecommendation(
                 strategy_type=GroupingStrategyType.FEATURE_BASED,
                 confidence=min(0.9, round(confidence, 2)),
                 rationale=(
                     f"Changes are distributed across {len(significant_dirs)} significant directories and involve {len(file_types)} file types. "
                     "This pattern often indicates cross-cutting features."
                 ),
                 estimated_pr_count=max(2, total_files // 6) # Estimate based on size/complexity
             ))

        # Module-based strategy (using file extensions as a proxy)
        # Condition: Distinct sets of file types, possibly low directory concentration
        if len(file_types) >= 2 and directory_concentration < 0.6:
             confidence = 0.5 + (len(file_types) * 0.05) + (0.6 - directory_concentration) * 0.2 # Base + bonus for types & low concentration
             recommendations.append(StrategyRecommendation(
                 strategy_type=GroupingStrategyType.MODULE_BASED,
                 confidence=min(0.85, round(confidence, 2)),
                 rationale=(
                     f"Changes include {len(file_types)} distinct file types ({', '.join(list(file_types.keys())[:3])}...) "
                     f"with relatively low directory concentration ({directory_concentration:.2f}). Grouping by module/type might be effective."
                 ),
                 estimated_pr_count=max(2, min(len(file_types), 4)) # Usually fewer PRs than directories
             ))

        # Size-balanced strategy (always an option, especially for large changes)
        if total_files > 15: # More relevant for larger changesets
             confidence = 0.4 + (min(total_files, 100) / 250) # Base confidence + bonus for size
             recommendations.append(StrategyRecommendation(
                 strategy_type=GroupingStrategyType.SIZE_BALANCED,
                 confidence=min(0.8, round(confidence, 2)), # Cap confidence as it's often a fallback
                 rationale=(
                     f"With {total_files} changed files, splitting into size-balanced PRs ensures manageable reviews, "
                     "regardless of logical structure."
                 ),
                 estimated_pr_count=max(2, total_files // 8) # Aim for roughly 8 files/PR average
             ))

        # Default or Mixed strategy if no clear winner
        if not recommendations or max(rec.confidence for rec in recommendations) < 0.65:
             recommendations.append(StrategyRecommendation(
                 strategy_type=GroupingStrategyType.MIXED, # Explicitly recommend MIXED
                 confidence=0.7, # Default confidence for MIXED
                 rationale=(
                     "No single strategy stands out strongly based on current metrics. "
                     "A mixed approach, potentially combining directory and feature/module aspects, is recommended."
                 ),
                 estimated_pr_count=max(2, min(total_files // 5, 7))
             ))


        # Sort recommendations by confidence (highest first)
        recommendations.sort(key=lambda x: x.confidence, reverse=True)

        # Select the highest confidence recommendation as the primary strategy
        primary_strategy_rec = recommendations[0] if recommendations else None
        primary_strategy_type = primary_strategy_rec.strategy_type if primary_strategy_rec else GroupingStrategyType.MIXED # Default if no recs

        # Create the explanation string
        explanation = (
            f"Selected '{primary_strategy_type.value}' as the primary strategy with "
            f"{primary_strategy_rec.confidence:.2f} confidence. "
            f"Rationale: {primary_strategy_rec.rationale}"
        ) if primary_strategy_rec else "No specific strategy strongly recommended; defaulting to MIXED."

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

        # Create and return the StrategySelection object
        strategy_selection_result = GroupingStrategyDecision(
            strategy_type=primary_strategy_type,
            recommendations=recommendations,
            repository_metrics=metrics_used,
            explanation=explanation
        )

        return strategy_selection_result