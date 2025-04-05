# main.py

import os
import sys
import argparse
from datetime import datetime
import shutil
from pathlib import Path # Use pathlib

# Assume configure_logging expects a 'verbose' boolean argument
from shared.utils.logging_utils import configure_logging, get_logger
from .crew import PRRecommendationCrew

logger = get_logger(__name__)


def main():
    """Main function for the PR recommendation system."""
    parser = argparse.ArgumentParser(description='Generate PR Grouping Recommendations.')
    parser.add_argument('repo_path', type=str, help='Path to the local git repository')
    parser.add_argument('--max-files', type=int, default=None, help='Maximum number of changed files to analyze fully')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Directory to save output files')
    parser.add_argument('--verbose', '-v', action='count', default=0, help='Increase verbosity level (e.g., -v, -vv)')
    args = parser.parse_args()

    # --- FIX IS HERE ---
    # Determine if verbose logging is enabled based on the count
    is_verbose = args.verbose > 0
    # Pass the boolean 'verbose' argument to configure_logging
    configure_logging(verbose=is_verbose)
    # --- END OF FIX ---

    # Determine crew verbosity level (0, 1, or 2)
    crew_verbose_level = min(args.verbose, 2) # Cap at 2 for crewAI

    # Validate repository path
    repo_path = Path(args.repo_path).resolve()
    if not (repo_path / '.git').is_dir():
        logger.error(f"Not a valid git repository: {repo_path}")
        sys.exit(1)

    # Ensure config files exist (using updated default content)
    config_dir = Path(__file__).parent / "config"
    ensure_config_files(config_dir)

    # Prepare output directory
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Outputs will be saved in: {output_dir}")

    try:
        logger.info(f"Initializing crew for repository: {repo_path}")

        # Create explicit inputs dictionary for kickoff
        inputs = {
            'repo_path': str(repo_path),  # Ensure str, not Path
            'use_summarization': True,
            'max_diff_size': 2000
        }
        
        # Add max_files if specified
        if args.max_files is not None:
            inputs['max_files'] = args.max_files
            logger.info(f"Setting max_files={args.max_files} in inputs")

        # Instantiate the crew with parsed arguments
        pr_crew_instance = PRRecommendationCrew(
            repo_path=str(repo_path),
            max_files=args.max_files,
            verbose=crew_verbose_level, # Pass the 0, 1, 2 level to crew
            output_dir=str(output_dir)
        )

        logger.info("Starting PR recommendation workflow...")
        logger.info(f"Using inputs: {inputs}")
        # Inputs for the kickoff are now handled by @before_kickoff
        result = pr_crew_instance.crew().kickoff(inputs=inputs)

        logger.info("----------------------------------------")
        logger.info("✅ PR Recommendation Workflow Completed")
        logger.info("----------------------------------------")
        logger.info(f"Final refined recommendations saved in: {output_dir}")
        # Log the final result structure (optional)
        if result:
             logger.debug(f"Final Crew Result (Output of last task):\n{result}")
        return 0

    except Exception as e:
        logger.error(f"❌ Error during PR recommendation workflow: {e}", exc_info=True)
        # exc_info=True automatically includes traceback in the log
        return 1

# ensure_config_files and create_default_configs remain the same as before

def ensure_config_files(config_dir: Path):
    """Ensure the YAML configuration files exist in the specified directory."""
    agents_yaml = config_dir / "agents.yaml"
    tasks_yaml = config_dir / "tasks.yaml"

    config_dir.mkdir(parents=True, exist_ok=True) # Ensure config dir exists

    if agents_yaml.exists() and tasks_yaml.exists():
        logger.debug("Configuration files found.")
        return

    logger.warning(f"Configuration files not found in {config_dir}. Creating defaults.")
    create_default_configs(config_dir)


def create_default_configs(config_dir: Path):
    """Create default agent and task configuration files."""

    # --- Updated Default Agents Config (NO TOOLS LIST) ---
    agents_content = """
# Agents Configuration for PR Recommendation Crew
pr_strategist:
  role: "Expert in organizing code changes into logical pull requests"
  goal: "Analyze repository changes, select the best grouping strategy, and generate initial PR groups based on the analysis"
  backstory: >
    You're a senior software architect with years of experience reviewing and organizing code changes.
    Your expertise lies in identifying patterns, technical dependencies, and logically grouping related
    changes together to create PRs that make sense. You meticulously analyze the state of the repository
    before deciding on the best path forward.
  allow_delegation: false
  verbose: true
  # memory: true # Consider enabling memory

pr_validator:
  role: "Quality assurance specialist for pull request organization"
  goal: "Validate and refine PR suggestions for completeness, coherence, and adherence to best practices"
  backstory: >
    You're a meticulous code reviewer and quality assurance specialist who understands what makes PRs
    reviewable and maintainable. You ensure PR groups are properly balanced, contain related changes,
    and follow engineering best practices. You take the initial suggestions and ensure they are polished
    and ready.
  allow_delegation: false
  verbose: true
"""

    # --- Tasks Config (remains the same) ---
    tasks_content = """
# Tasks Configuration for PR Recommendation Crew
# Note: Agent assignment and tools are handled in crew.py when using @CrewBase decorators.
# This file primarily provides descriptions and expected outputs for CrewBase to use.

# Task 1: Analyze Repository
analyze_repository_changes:
  description: >
    Analyze the git repository located at '{repo_path}' to identify all outstanding
    file changes (staged and unstaged). Use the 'repo_analyzer' tool.
    Pass 'max_files' from inputs if provided.
  expected_output: >
    A comprehensive RepositoryAnalysis object containing a list of FileChange objects,
    directory summaries, and total change counts.

# Task 2: Calculate Metrics
calculate_repository_metrics:
  description: >
    Using the RepositoryAnalysis from the previous step (analyze_repository_changes),
    calculate objective metrics about the changes using the 'repo_metrics' tool.
    Focus on directory structure, file types, and change sizes.
  expected_output: >
    A RepositoryMetrics object containing various calculated metrics.

# Task 3: Analyze Patterns
analyze_change_patterns:
  description: >
    Using the RepositoryAnalysis from the first step (analyze_repository_changes),
    analyze file names for common patterns, similarities, and relationships
    using the 'pattern_analyzer' tool.
  expected_output: >
    A PatternAnalysisResult object detailing identified patterns.

# Optional Task 3b: Analyze Directory Structure
# analyze_directory_structure:
#   description: >
#     Using the RepositoryAnalysis from the first step (analyze_repository_changes),
#     perform an in-depth analysis of the directory structure, hierarchy, and complexity
#     using the 'directory_analyzer' tool.
#   expected_output: >
#     A DirectoryAnalysisResult object with detailed structure analysis.

# Task 4: Select Strategy
select_grouping_strategy:
  description: >
    Evaluate the RepositoryAnalysis (from analyze_repository_changes) and calculated
    RepositoryMetrics (from calculate_repository_metrics) to decide the most suitable
    high-level grouping strategy (e.g., directory_based, feature_based) using the
    'grouping_strategy_selector' tool. Provide rationale and confidence rankings.
  expected_output: >
    A GroupingStrategyDecision object containing the selected primary strategy type,
    recommendations, influencing metrics, and rationale.

# Task 5: Generate Groups
generate_pr_groups:
  description: >
    Apply the selected grouping strategy type (from the 'strategy_type' field in the
    GroupingStrategyDecision of the previous step) to the file changes identified in
    RepositoryAnalysis (from analyze_repository_changes). Use RepositoryMetrics (from
    calculate_repository_metrics) and PatternAnalysisResult (from analyze_change_patterns)
    as context. Use the 'file_grouper' tool to create specific PR groups with titles,
    rationales, and suggested metadata.
  expected_output: >
    A PRGroupingStrategy object populated with a list of PRGroup objects, detailing the proposed pull requests.

# Task 6: Validate Groups
validate_pr_groups:
  description: >
    Validate the initial PR groups generated in the previous step (generate_pr_groups)
    using the original RepositoryAnalysis (from analyze_repository_changes) as context.
    Use the 'group_validator' tool to check for best practices like completeness,
    balance, coherence, clarity, and related file pairings.
  expected_output: >
    A PRValidationResult object indicating validity and listing any identified issues.

# Task 7: Refine Groups
refine_pr_groups:
  description: >
    Review the validation results (from validate_pr_groups) and the initial PR groups
    (from generate_pr_groups). If validation issues were found, use the 'group_refiner'
    tool to refine the groups by addressing issues like imbalance, duplication, missing files,
    or separated concerns. Ensure all original files are accounted for.
  expected_output: >
    A final, refined PRGroupingStrategy object containing the polished list of PRGroup objects.
"""

    try:
        with open(config_dir / "agents.yaml", 'w') as f:
            f.write(agents_content.strip())
        with open(config_dir / "tasks.yaml", 'w') as f:
            f.write(tasks_content.strip())
        logger.info(f"Created default configuration files in {config_dir}")
    except Exception as e:
        logger.error(f"Error creating default configuration files: {e}")


if __name__ == "__main__":
    sys.exit(main())