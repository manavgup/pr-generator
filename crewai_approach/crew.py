# crew.py
import json
import os
from pathlib import Path
from typing import Optional, Dict, List # Added List

from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, task, crew, before_kickoff
from shared.utils.logging_utils import get_logger
# --- Model Imports ---
# (Keep previous imports for models)
from .models.agent_models import (
    RepositoryMetrics, PatternAnalysisResult, GroupingStrategyDecision,
    PRGroupingStrategy, PRValidationResult, DirectoryAnalysisResult
)
from shared.models.analysis_models import RepositoryAnalysis

# --- Tool Imports ---
# (Keep previous imports for tools)
from .tools.repo_analyzer import RepoAnalyzerTool
from .tools.repo_metrics import RepositoryMetricsCalculator
from .tools.directory_analyzer import DirectoryAnalyzer
from .tools.pattern_analyzer import PatternAnalyzerTool
from .tools.grouping_strategy_selector import GroupingStrategySelector
from .tools.file_grouper import FileGrouperTool
from .tools.group_validator import GroupValidatorTool
from .tools.group_refiner import GroupRefiner

logger = get_logger(__name__)

@CrewBase
class PRRecommendationCrew:
    """
    Crew that analyzes repository changes and suggests logical PR groupings.
    """
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self,
                 repo_path: str,
                 max_files: Optional[int] = None,
                 verbose: bool | int = False, # Allow int for crewAI verbosity levels
                 output_dir: str = 'outputs'):
        """
        Initialize the PR Recommendation crew.

        Args:
            repo_path: Path to the git repository.
            max_files: Maximum number of files to analyze.
            verbose: Verbosity level (True/False or 1/2).
            output_dir: Directory to save intermediate and final outputs.
        """
        logger.info(f"Initializing PRRecommendationCrew for repository: {repo_path}")
        logger.info(f"Max files: {max_files}, Verbose: {verbose}, Output Dir: {output_dir}")

        self.repo_path = Path(repo_path).resolve() # Ensure absolute path
        self.max_files = max_files
        self.verbose = verbose
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True) # Ensure output dir exists

        # --- Tool Instantiation ---
        logger.info("Instantiating tools...")
        self.repo_analyzer_tool = RepoAnalyzerTool()
        self.repo_metrics_tool = RepositoryMetricsCalculator()
        self.directory_analyzer_tool = DirectoryAnalyzer()
        self.pattern_analyzer_tool = PatternAnalyzerTool()
        self.grouping_strategy_selector_tool = GroupingStrategySelector()
        self.file_grouper_tool = FileGrouperTool()
        self.group_validator_tool = GroupValidatorTool()
        self.group_refiner_tool = GroupRefiner()
        # Create a map for easy access by name if needed elsewhere, though direct access is fine too
        self.tools_map = {
            "repo_analyzer": self.repo_analyzer_tool,
            "repo_metrics": self.repo_metrics_tool,
            "directory_analyzer": self.directory_analyzer_tool,
            "pattern_analyzer": self.pattern_analyzer_tool,
            "grouping_strategy_selector": self.grouping_strategy_selector_tool,
            "file_grouper": self.file_grouper_tool,
            "group_validator": self.group_validator_tool,
            "group_refiner": self.group_refiner_tool
        }
        logger.info("Tools instantiated.")

        # Simplified check - GitOperations performs deeper checks
        if not (self.repo_path / ".git").is_dir():
             raise ValueError(f"Provided path is not a valid Git repository: {self.repo_path}")


    @before_kickoff
    def prepare_inputs(self, inputs: Optional[Dict] = None) -> Dict:
        """Prepare inputs before the crew starts."""
        logger.info("Preparing inputs for crew kickoff")
        inputs = inputs or {}
        
        # Always include max_files even if None
        inputs.update({
            'repo_path': str(self.repo_path),
            'max_files': self.max_files,  # This will be None if not provided
            'use_summarization': True,
            'max_diff_size': 2000
        })
        
        logger.debug(f"Inputs prepared: {inputs}")
        return inputs

    def _save_output_callback(self, step_name: str):
        """Generic callback to save task output, handling Path objects."""
        filepath = self.output_dir / f"{step_name}.json"
        def save_output(output):
            logger.info(f"Saving output for step '{step_name}' to {filepath}")
            try:
                # Prepare data for JSON serialization: convert Path objects to strings
                if hasattr(output, 'model_dump'):
                    # Use Pydantic's model_dump for robust serialization
                    # By default, model_dump handles Path objects correctly if using Pydantic v2+
                    # Let's ensure it produces a dict first
                    data_to_save = output.model_dump(mode='json')
                elif isinstance(output, (dict, list)):
                     # If it's already a dict/list, we might need to manually convert Paths deep within
                     # This requires a recursive function - more complex. Let's hope Pydantic handles it.
                     # For simplicity, we assume Pydantic output first. If errors persist here,
                     # we might need a recursive path-to-string converter.
                     logger.warning(f"Output for step '{step_name}' is dict/list, manual Path conversion might be needed if serialization fails.")
                     data_to_save = output # Try saving directly first
                else:
                     # Fallback for non-Pydantic, non-dict/list outputs
                     data_to_save = str(output)

                # Now serialize the processed data
                with open(filepath, 'w') as f:
                    # Use json.dump with default=str as a fallback for any other non-serializable types
                    json.dump(data_to_save, f, indent=2, default=str)

                logger.info(f"Successfully saved output for '{step_name}'.")

            except TypeError as te:
                # Specifically catch TypeError related to JSON serialization
                 logger.error(f"JSON Serialization TypeError for step '{step_name}': {te}. Attempting fallback with default=str.")
                 # Attempt fallback serialization, converting unknown types to string
                 try:
                      with open(filepath, 'w') as f:
                           json.dump(output, f, indent=2, default=str) # Force conversion of non-serializable types
                      logger.info(f"Successfully saved output for '{step_name}' using fallback serialization.")
                 except Exception as e_fallback:
                      logger.error(f"Fallback serialization also failed for step '{step_name}': {e_fallback}. Output type was: {type(output)}")

            except Exception as e:
                 # Catch other potential errors during saving
                 logger.error(f"Failed to save output for step '{step_name}': {e}. Output type was: {type(output)}")

            return output # Must return the original output for crewAI
        return save_output

    # --- Agent Definitions using @agent ---

    @agent
    def pr_strategist(self) -> Agent:
        """Creates the PR Strategist agent."""
        agent_id = "pr_strategist" # Match key in agents.yaml
        logger.info(f"Creating agent: {agent_id}")
        agent_config = self.agents_config[agent_id]
        logger.debug(f"{agent_id} config: {agent_config}")

        # *** Define the tools for this agent explicitly here ***
        strategist_tools = [
            self.repo_analyzer_tool,
            self.repo_metrics_tool,
            self.pattern_analyzer_tool,
            self.grouping_strategy_selector_tool,
            self.file_grouper_tool,
            # self.directory_analyzer_tool, # Add if used by its tasks
        ]

        return Agent(
            config=agent_config,          # Pass the loaded config dictionary
            tools=strategist_tools,  # Assign the actual tool objects
            verbose=self.verbose,         # Set verbosity from crew initialization
            allow_delegation=agent_config.get('allow_delegation', False), # Ensure delegation is set
            # memory=agent_config.get('memory', False) # Optionally configure memory
            # llm=self.some_llm_instance # Optionally override LLM
        )

    @agent
    def pr_validator(self) -> Agent:
        """Creates the PR Validator agent."""
        agent_id = "pr_validator" # Match key in agents.yaml
        logger.info(f"Creating agent: {agent_id}")
        agent_config = self.agents_config[agent_id]
        logger.debug(f"{agent_id} config: {agent_config}")

        # *** Define the tools for this agent explicitly here ***
        validator_tools = [
            self.group_validator_tool,
            self.group_refiner_tool,
        ]

        return Agent(
            config=agent_config,
            tools=validator_tools,
            verbose=self.verbose,
            allow_delegation=agent_config.get('allow_delegation', False),
            # memory=agent_config.get('memory', False)
        )

    # --- Task Definitions (Using correct pattern) ---

    @task
    def analyze_repository_changes(self) -> Task:
        """Creates the 'analyze_repository_changes' task using config."""
        task_id = "analyze_repository_changes"
        logger.info(f"Creating task: {task_id}")
        return Task(
            config=self.tasks_config[task_id],
            agent=self.pr_strategist(), # Assign agent instance via method call
            tools=[self.repo_analyzer_tool],
            output_pydantic=RepositoryAnalysis,
            callback=self._save_output_callback("step_1_analysis")
        )

    @task
    def calculate_repository_metrics(self) -> Task:
        """Creates the 'calculate_repository_metrics' task using config."""
        task_id = "calculate_repository_metrics"
        logger.info(f"Creating task: {task_id}")
        return Task(
            config=self.tasks_config[task_id],
            agent=self.pr_strategist(), # Correct agent
            tools=[self.repo_metrics_tool],
            context=[self.analyze_repository_changes()],
            output_pydantic=RepositoryMetrics,
            callback=self._save_output_callback("step_2_metrics")
        )

    @task
    def analyze_change_patterns(self) -> Task:
        """Creates the 'analyze_change_patterns' task using config."""
        task_id = "analyze_change_patterns"
        logger.info(f"Creating task: {task_id}")
        return Task(
            config=self.tasks_config[task_id],
            agent=self.pr_strategist(), # Correct agent
            tools=[self.pattern_analyzer_tool],
            context=[self.analyze_repository_changes()],
            output_pydantic=PatternAnalysisResult,
            callback=self._save_output_callback("step_3_patterns")
        )

    # Optional: Task for Directory Analysis
    # @task
    # def analyze_directory_structure(self) -> Task:
    #      task_id = "analyze_directory_structure" ...

    @task
    def select_grouping_strategy(self) -> Task:
        """Creates the 'select_grouping_strategy' task using config."""
        task_id = "select_grouping_strategy"
        logger.info(f"Creating task: {task_id}")
        return Task(
            config=self.tasks_config[task_id],
            agent=self.pr_strategist(), # Correct agent
            tools=[self.grouping_strategy_selector_tool],
            context=[
                self.analyze_repository_changes(),
                self.calculate_repository_metrics(),
                # Add context if needed by selector logic or agent prompt
                # self.analyze_change_patterns(),
            ],
            output_pydantic=GroupingStrategyDecision,
            callback=self._save_output_callback("step_4_strategy_decision")
        )

    @task
    def generate_pr_groups(self) -> Task:
        """Creates the 'generate_pr_groups' task using config."""
        task_id = "generate_pr_groups"
        logger.info(f"Creating task: {task_id}")
        return Task(
            config=self.tasks_config[task_id],
            agent=self.pr_strategist(), # Correct agent
            tools=[self.file_grouper_tool],
            context=[
                self.analyze_repository_changes(),
                self.calculate_repository_metrics(),
                self.analyze_change_patterns(),
                self.select_grouping_strategy(),
            ],
            output_pydantic=PRGroupingStrategy,
            callback=self._save_output_callback("step_5_initial_groups")
        )

    @task
    def validate_pr_groups(self) -> Task:
        """Creates the 'validate_pr_groups' task using config."""
        task_id = "validate_pr_groups"
        logger.info(f"Creating task: {task_id}")
        return Task(
            config=self.tasks_config[task_id],
            agent=self.pr_validator(), # Assign agent instance via method call
            tools=[self.group_validator_tool],
            context=[
                self.analyze_repository_changes(),
                self.generate_pr_groups()
            ],
            output_pydantic=PRValidationResult,
            callback=self._save_output_callback("step_6_validation")
        )

    @task
    def refine_pr_groups(self) -> Task:
        """Creates the 'refine_pr_groups' task using config."""
        task_id = "refine_pr_groups"
        logger.info(f"Creating task: {task_id}")
        final_output_filename = f"{Path(self.output_dir).name}_final_recommendations" # Cleaner filename
        return Task(
            config=self.tasks_config[task_id],
            agent=self.pr_validator(), # Correct agent
            tools=[self.group_refiner_tool],
            context=[
                self.generate_pr_groups(),
                self.validate_pr_groups()
            ],
            output_pydantic=PRGroupingStrategy,
            callback=self._save_output_callback(final_output_filename)
        )

    # --- Crew Definition ---
    @crew
    def crew(self) -> Crew:
        """Creates the PR Recommendation crew."""
        logger.info("Assembling the PR Recommendation crew...")
        return Crew(
            agents=[ # Provide agent instances from the @agent methods
                self.pr_strategist(),
                self.pr_validator()
            ],
            tasks=[ # Define the sequential flow by calling the @task methods
                self.analyze_repository_changes(),
                self.calculate_repository_metrics(),
                self.analyze_change_patterns(),
                # self.analyze_directory_structure(), # Optional task
                self.select_grouping_strategy(),
                self.generate_pr_groups(),
                self.validate_pr_groups(),
                self.refine_pr_groups(),
            ],
            process=Process.sequential,
            verbose=self.verbose,
            # memory=True,
            # cache=True
        )