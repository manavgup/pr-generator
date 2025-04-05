"""
Models for the PR Recommendation System agents and tools.
"""
from enum import Enum
from typing import List, Dict, Optional, Any, Set
from pathlib import Path
from pydantic import BaseModel, Field, computed_field

class GroupingStrategyType(str, Enum):
    """Type of grouping strategy for PR recommendations."""
    DIRECTORY_BASED = "directory_based"
    FEATURE_BASED = "feature_based"
    MODULE_BASED = "module_based"
    SIZE_BALANCED = "size_balanced"
    MIXED = "mixed"

class DirectoryComplexity(BaseModel):
    """Represents complexity analysis for a directory."""
    path: str = Field(..., description="Directory path")
    file_count: int = Field(default=0, description="Number of files in this directory")
    changed_file_count: int = Field(default=0, description="Number of changed files in this directory")
    extension_counts: Dict[str, int] = Field(default_factory=dict, description="Count of file extensions in this directory")
    estimated_complexity: float = Field(default=0.0, description="Estimated complexity score (0-10)")

class StrategyRecommendation(BaseModel):
    """Represents a recommended grouping strategy with rationale."""
    strategy_type: GroupingStrategyType = Field(..., description="Recommended strategy type")
    confidence: float = Field(..., description="Confidence score for this recommendation (0-1)")
    rationale: str = Field(..., description="Explanation of why this strategy is recommended")
    estimated_pr_count: int = Field(default=1, description="Estimated number of PRs this strategy would generate")

class NamingPattern(BaseModel):
    """Represents a naming pattern detected in files."""
    pattern: str = Field(..., description="Pattern expression (e.g., 'test_*.py')")
    matches: List[str] = Field(default_factory=list, description="List of file names matching this pattern")
    type: str = Field(..., description="Type of pattern (e.g., 'test_files')")
    description: str = Field(..., description="Human-readable description of the pattern")

class SimilarNameGroup(BaseModel):
    """Represents a group of files with similar names."""
    base_pattern: str = Field(..., description="Common pattern or root")
    files: List[str] = Field(default_factory=list, description="List of file paths with similar names")

class CommonPatternGroup(BaseModel):
    """Represents files sharing a common prefix or suffix."""
    pattern_type: str = Field(..., description="Type of pattern ('prefix' or 'suffix')")
    pattern_value: str = Field(..., description="The prefix or suffix value")
    files: List[str] = Field(default_factory=list, description="List of file paths with this pattern")

class FilePair(BaseModel):
    """Represents a pair of related files."""
    file1: str = Field(..., description="Path to the first file")
    file2: str = Field(..., description="Path to the second file")
    relation_type: str = Field(..., description="Type of relationship (e.g., 'implementation_test')")
    base_name: Optional[str] = Field(None, description="Common base name if applicable")

class RelatedFileGroup(BaseModel):
    """Represents a group of related files."""
    type: str = Field(..., description="Type of relation group (e.g., 'implementation_test_pairs')")
    pairs: List[FilePair] = Field(default_factory=list, description="List of file pairs in this relation")

class CommonPatterns(BaseModel):
    """Represents common patterns found in file names."""
    common_prefixes: List[CommonPatternGroup] = Field(default_factory=list, description="Common prefixes found in files")
    common_suffixes: List[CommonPatternGroup] = Field(default_factory=list, description="Common suffixes found in files")

class PatternAnalysisResult(BaseModel):
    """Results of pattern analysis on a repository."""
    naming_patterns: List[NamingPattern] = Field(default_factory=list, description="Naming patterns detected")
    similar_names: List[SimilarNameGroup] = Field(default_factory=list, description="Groups of files with similar names")
    common_patterns: CommonPatterns = Field(default_factory=CommonPatterns, description="Common patterns in file names")
    related_files: List[RelatedFileGroup] = Field(default_factory=list, description="Groups of related files")
    analysis_summary: str = Field("", description="Summary of the pattern analysis")
    confidence: float = Field(default=0.0, description="Confidence in the pattern analysis (0-1)")

class PRGroup(BaseModel):
    """Represents a group of files for a potential PR."""
    title: str = Field(..., description="PR title")
    files: List[str] = Field(default_factory=list, description="List of file paths in this group")
    rationale: str = Field(..., description="Explanation for grouping these files")
    estimated_size: int = Field(default=0, description="Estimated size/complexity of this PR")
    directory_focus: Optional[str] = Field(None, description="Primary directory this group focuses on")
    feature_focus: Optional[str] = Field(None, description="Feature or functionality this group focuses on")
    suggested_branch_name: Optional[str] = Field(None, description="Suggested git branch name for this PR")
    suggested_pr_description: Optional[str] = Field(None, description="Suggested PR description")

class PRGroupingStrategy(BaseModel):
    """Represents a strategy for grouping files into PRs."""
    strategy_type: GroupingStrategyType = Field(..., description="Type of grouping strategy used")
    groups: List[PRGroup] = Field(default_factory=list, description="List of PR groups generated by this strategy")
    explanation: str = Field(..., description="Explanation of the grouping results (not the selection rationale)")
    estimated_review_complexity: float = Field(default=5.0, description="Estimated review complexity (1-10) of the generated groups")
    ungrouped_files: List[str] = Field(default_factory=list, description="Files that couldn't be grouped by this strategy")


class GroupValidationIssue(BaseModel):
    """Represents an issue found during PR group validation."""
    severity: str = Field(..., description="Issue severity (high, medium, low)")
    issue_type: str = Field(..., description="Type of issue")
    description: str = Field(..., description="Description of the issue")
    affected_groups: List[str] = Field(default_factory=list, description="Group titles affected by this issue")
    recommendation: str = Field(..., description="Recommendation for fixing the issue")

class PRValidationResult(BaseModel):
    """Results of validating a PR grouping strategy."""
    is_valid: bool = Field(..., description="Whether the grouping strategy is valid")
    issues: List[GroupValidationIssue] = Field(default_factory=list, description="List of validation issues")
    validation_notes: str = Field(..., description="Notes about the validation results")
    strategy_type: GroupingStrategyType = Field(..., description="Type of grouping strategy validated")

class RepositoryMetrics(BaseModel):
    """
    Model representing objective metrics about a repository's changes.
    """
    repo_path: str = Field(..., description="Path to the git repository.")
    total_files_changed: int = Field(..., description="Total number of files changed.")
    total_lines_changed: int = Field(..., description="Total number of lines changed.")
    directory_metrics: Dict[str, Any] = Field(..., description="Metrics related to directory structure.")
    file_type_metrics: Dict[str, Any] = Field(..., description="Metrics related to file types.")
    change_metrics: Dict[str, Any] = Field(..., description="Metrics related to change patterns.")
    complexity_indicators: List[str] = Field(..., description="List of complexity indicator strings.")


class ParentChildRelation(BaseModel):
    """Represents a parent-child directory relationship."""
    parent: str = Field(..., description="Path of the parent directory")
    child: str = Field(..., description="Path of the child directory")

class PotentialFeatureDirectory(BaseModel):
    """Represents a directory identified as potentially feature-related."""
    directory: str = Field(..., description="Path of the directory")
    is_diverse: bool = Field(..., description="Indicates if the directory contains diverse file types")
    is_cross_cutting: bool = Field(..., description="Indicates if the directory is related to multiple others")
    file_types: List[str] = Field(..., description="List of file extensions found in the directory")
    related_directories: List[str] = Field(..., description="List of related directory paths")
    confidence: float = Field(..., description="Confidence score (0-1) that this represents a feature")

class DirectoryAnalysisResult(BaseModel):
    """Results of analyzing the directory structure of changes."""
    directory_count: int = Field(..., description="Total number of directories with changes")
    max_depth: int = Field(..., description="Maximum depth of changed directories in the hierarchy")
    avg_files_per_directory: float = Field(..., description="Average number of changed files per directory")
    directory_complexity: List[DirectoryComplexity] = Field(..., description="Complexity analysis for each directory")
    parent_child_relationships: List[ParentChildRelation] = Field(..., description="Identified parent-child relationships between changed directories")
    potential_feature_directories: List[PotentialFeatureDirectory] = Field(..., description="Directories identified as potentially feature-related")

class GroupingStrategyDecision(BaseModel):
    """Represents the selected grouping strategy and the rationale behind it."""
    strategy_type: GroupingStrategyType = Field(..., description="The primary grouping strategy selected")
    recommendations: List[StrategyRecommendation] = Field(..., description="List of all considered strategies, ranked by confidence")
    repository_metrics: Dict[str, Any] = Field(..., description="Key repository metrics used for the selection decision")
    explanation: str = Field(..., description="Summary explanation for the chosen strategy")