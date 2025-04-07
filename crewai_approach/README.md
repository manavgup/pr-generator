# PR Recommendation System - CrewAI Approach

This module uses the CrewAI framework to analyze git repository changes and recommend logical pull request groupings. It follows a structured workflow involving analysis, strategy selection, grouping, validation, and refinement.

## Architecture

The system uses a CrewAI-based architecture following the recommended decorator pattern (`@CrewBase`) with YAML configuration:

1.  **Agents**: Defined in `config/agents.yaml`.
    *   **PR Strategist**: Analyzes repository structure, selects a grouping strategy, and generates initial PR groups.
    *   **PR Validator**: Validates the initial PR groups against best practices and refines them if necessary.

2.  **Tasks**: Defined in `config/tasks.yaml`. The workflow follows these sequential steps:
    *   **analyze\_repository\_changes**: Analyzes the git status to identify changed files.
    *   **calculate\_repository\_metrics**: Computes objective metrics about the changes (size, distribution, etc.).
    *   **analyze\_change\_patterns**: Identifies naming patterns and file relationships.
    *   **(Optional) analyze\_directory\_structure**: Performs deeper analysis of directory organization.
    *   **select\_grouping\_strategy**: Chooses the best high-level strategy (e.g., directory, feature) based on analysis.
    *   **generate\_pr\_groups**: Applies the selected strategy to group files into specific PRs.
    *   **validate\_pr\_groups**: Checks the generated groups for issues (size, completeness, coherence).
    *   **refine\_pr\_groups**: Adjusts the groups based on validation feedback.

3.  **Tools**: Specialized Python classes inheriting from `crewai.tools.BaseTool`, performing specific operations:
    *   **RepoAnalyzerTool**: Analyzes git repository status.
    *   **RepositoryMetricsCalculator**: Calculates metrics from repository analysis.
    *   **PatternAnalyzerTool**: Identifies file patterns.
    *   **DirectoryAnalyzer**: Analyzes directory structure (used by an optional task).
    *   **GroupingStrategySelector**: Selects the high-level grouping strategy.
    *   **FileGrouperTool**: Groups files based on the selected strategy.
    *   **GroupValidatorTool**: Validates PR groups against rules.
    *   **GroupRefiner**: Modifies PR groups based on validation issues.

4.  **Models**: Pydantic models define the data structures passed between tasks:
    *   **Shared Models**: (`shared/models/`) Define core concepts like `RepositoryAnalysis`, `FileChange`, `DirectorySummary`. Reusable across frameworks.
    *   **Agent Models**: (`models/agent_models.py`) Define specific inputs/outputs for the CrewAI workflow, like `GroupingStrategyDecision`, `PRGroupingStrategy`, `PRValidationResult`.

5.  **Data Format**: Tools in this system use a standard pattern to handle data:
    *   **Input**: Tools accept dictionary inputs (rather than direct Pydantic models) for flexibility
    *   **Output**: Tools return JSON strings (rather than Pydantic objects) for reliable serialization
    *   **Validation**: Tools perform explicit validation of input data and return structured error responses
    *   This approach ensures compatibility with CrewAI's data passing mechanisms and provides robust error handling.

## Usage

### Command Line

Run the PR Recommendation System from the command line:

```bash
# Ensure you are in the directory containing the main module or adjust path
python -m your_module_name.main path/to/your/repo --verbose 2
```

(Replace your_module_name with the actual name of the directory containing main.py and crew.py, e.g., crewai_approach)

Command-line options:
    --repo_path: Path to the local git repository (required positional argument).
    --max-files: Maximum number of changed files to analyze fully (optional).
    --output-dir: Directory to save output files (default: ./outputs).
    --verbose, -v: Increase verbosity level (use -v for agent logs, -vv for detailed agent/tool logs).

## Programmatic Usage

Use the system programmatically:
```
from your_module_name.crew import PRRecommendationCrew # Adjust import path

# Configuration
repo_path = "/path/to/your/repo"
output_directory = "pr_run_outputs"
verbosity_level = 1 # 0: Normal, 1: Agent logs, 2: Agent + Tool logs

# Create the crew instance
pr_crew = PRRecommendationCrew(
    repo_path=repo_path,
    max_files=150,        # Optional: Limit files analyzed
    verbose=verbosity_level,
    output_dir=output_directory
)

# Run the crew (inputs are handled by @before_kickoff)
# The result will be the output of the final task (refined PRGroupingStrategy)
final_result = pr_crew.crew().kickoff()

# Process the final_result (which is a PRGroupingStrategy object)
if final_result:
    print(f"Successfully generated {len(final_result.groups)} PR group recommendations.")
    # Access final_result.groups, final_result.explanation etc.
    # The final JSON is also saved in the specified output directory.
```
(Replace your_module_name with the actual import path)

## Observability and Debugging
The system includes several features to help with observability and debugging:

- Task Outputs: Each task's output is saved as a JSON file in the output directory, allowing you to inspect intermediate results.
- Logging: Comprehensive logging at different verbosity levels helps track the execution flow and identify issues.
- AgentOps Integration: The system can be monitored using AgentOps, which provides a waterfall view of execution spans, clearly showing agent reasoning and tool usage.
- Error Handling: Tools include robust error handling and return structured error responses as JSON, making it easier to diagnose issues.

## Configuration

The system uses YAML files for agent and task configurations, managed by CrewBase:

    config/agents.yaml: Defines agent roles, goals, backstories, and the superset of tools they can use.

    config/tasks.yaml: Defines task descriptions and expected outputs. Agent and specific tool assignments for a task are handled in crew.py using decorators.

## Agent Configuration (agents.yaml)
```
# Agent definitions used by CrewBase
# Agent definitions used by CrewBase
pr_strategist:
  role: "Expert in organizing code changes..."
  goal: "Analyze repository changes, select strategy, generate initial groups..."
  backstory: "You're a senior software architect..."
  tools: # Tools this agent is allowed to use across its tasks
    - repo_analyzer
    - repo_metrics
    - pattern_analyzer
    - grouping_strategy_selector
    - file_grouper
    # - directory_analyzer
  allow_delegation: false
  # verbose controlled by crew setting

pr_validator:
  role: "Quality assurance specialist..."
  goal: "Validate and refine PR suggestions..."
  backstory: "You're a meticulous code reviewer..."
  tools:
    - group_validator
    - group_refiner
  allow_delegation: false
```

## Task Configuration (tasks.yaml)

Provides descriptive context for each task defined in crew.py:

```
# Task descriptions loaded by CrewBase
analyze_repository_changes:
  description: "Analyze the git repository located at '{repo_path}'..."
  expected_output: "A comprehensive RepositoryAnalysis object..."

calculate_repository_metrics:
  description: "Using the RepositoryAnalysis from the previous step..."
  expected_output: "A RepositoryMetrics object containing..."

# ... entries for all 7 tasks ...

refine_pr_groups:
  description: "Review the validation results (from validate_pr_groups)..."
  expected_output: "A final, refined PRGroupingStrategy object..."
```

## Output

The primary output is the result of the final refine_pr_groups task, which is a PRGroupingStrategy object. This object contains the list of recommended PRGroup objects.

Intermediate outputs for each step are saved as JSON files in the specified --output-dir (default: outputs/).

The final refined recommendations are saved to a file named <output_dir_name>_final_recommendations.json within the output directory.

Example structure of a PRGroup within the final PRGroupingStrategy.groups list:
```
{
  "title": "Refactor Authentication Logic",
  "files": [
    "src/auth/service.py",
    "src/auth/models.py",
    "tests/auth/test_service.py"
  ],
  "rationale": "Groups changes related to the core authentication service, including models and tests.",
  "estimated_size": 3,
  "directory_focus": "src/auth",
  "feature_focus": "authentication",
  "suggested_branch_name": "feature/refactor-auth-logic",
  "suggested_pr_description": "## Refactor Authentication Logic\n\nThis PR refactors the main authentication service...\n\n## Files Changed\n\n- `src/auth/service.py`\n- `src/auth/models.py`\n- `tests/auth/test_service.py`\n"
}
```

## Development

To extend or modify the system:

    Edit config/agents.yaml / config/tasks.yaml: Modify agent personalities or task descriptions.

    Modify crew.py: Adjust the task sequence, context passing, tool assignments per task, or agent instantiation logic (@agent, @task methods).

    Add/Modify Tools: Create or update tool classes (inheriting from BaseTool) in the tools/ directory. Remember to instantiate and assign them in crew.py.

    Add/Modify Models: Update Pydantic models in models/ or shared/models/. Ensure dependent tools and tasks are updated.

### Configuration Auto-Creation

If config/agents.yaml or config/tasks.yaml are missing, the system will create default versions upon running main.py, reflecting the 7-task structure defined in crew.py.

## Tool Design Principles

When creating or modifying tools for this system, follow these design principles:

1. **Input Simplicity**: Tools accept dictionaries rather than Pydantic models directly

```python
def _run(self, repo_path: str, repository_analysis: Dict[str, Any], **kwargs) -> str:
```

2. **Output Consistency**: Tools return JSON strings (serialized dictionaries) rather than objects
```python
return json.dumps(result, indent=2)
```

3. **Explicit Validation**: Tools validate inputs and handle errors gracefully

```python
try:
    # Extract and validate data
except ValidationError as e:
    error_msg = f"Validation error: {str(e)}"
    logger.error(error_msg)
    return json.dumps({"error": error_msg})
```

4. **Robust Error Handling**: Tools return well-structured error responses that won't break the pipeline
```python
except Exception as e:
    error_msg = f"Error analyzing patterns: {str(e)}"
    logger.error(error_msg)
    return json.dumps({
        "error": error_msg,
        "naming_patterns": [],
        # ... minimal valid structure ...
    })
```
These principles ensure tools work reliably within the CrewAI framework, where serialization and data passing between components are critical concerns.
