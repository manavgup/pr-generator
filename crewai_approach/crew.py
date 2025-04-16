# crew.py
import json
import re
from pathlib import Path
from typing import Optional, Dict, List

from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, task, crew, before_kickoff
from shared.utils.logging_utils import get_logger
# --- Model Imports ---
from .models.agent_models import (
    RepositoryMetrics, PatternAnalysisResult, GroupingStrategyDecision,
    PRGroupingStrategy, PRValidationResult, DirectoryAnalysisResult
)
from models.batching_models import BatchSplitterOutput
from shared.models.analysis_models import RepositoryAnalysis

# --- Tool Imports ---
from .tools.repo_analyzer_tool import RepoAnalyzerTool
from .tools.repo_metrics_tool import RepositoryMetricsCalculator
from .tools.pattern_analyzer_tool import PatternAnalyzerTool
from .tools.grouping_strategy_selector_tool import GroupingStrategySelector
# Batching Tools
from .tools.batch_splitter_tool import BatchSplitterTool
from .tools.group_merging_tool import GroupMergingTool
# Worker Tools
from .tools.file_grouper_tool import FileGrouperTool
from .tools.group_validator_tool import GroupValidatorTool
from .tools.group_refiner_tool import GroupRefinerTool

logger = get_logger(__name__)

@CrewBase
class HierarchicalPRCrew:
    """
    Hierarchical Crew using Process.hierarchical to analyze repository changes
    and suggest logical PR groupings, handling large repositories via batching.
    """
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self,
                 repo_path: str,
                 max_files: Optional[int] = None,
                 max_batch_size: int = 50,
                 verbose: bool | int = False, # Allow int for crewAI verbosity levels
                 output_dir: str = 'outputs',
                 manager_llm_name: str = "gpt-4o"):
        """
        Initialize the PR Recommendation crew.

        Args:
            repo_path: Path to the git repository.
            max_files: Maximum number of files to analyze.
            verbose: Verbosity level (True/False or 1/2).
            output_dir: Directory to save intermediate and final outputs.
        """
        logger.info(f"Initializing HierarchicalPRCrew (Hierarchical Process) for repository: {repo_path}")
        logger.info(f"Max files: {max_files}, Max Batch Size: {max_batch_size}, Verbose: {verbose}")

        self.repo_path = Path(repo_path).resolve() # Ensure absolute path
        self.max_files = max_files
        self.max_batch_size = max_batch_size
        self.verbose = verbose
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True) # Ensure output dir exists
        self.manager_llm_name = manager_llm_name

        # Check repo path
        if not (self.repo_path / ".git").is_dir():
            raise ValueError(f"Not a valid Git repository: {self.repo_path}")

        # --- Tool Instantiation ---
        logger.info("Instantiating tools...")
        # Ensure all these tools get the repo_path parameter:
        self.repo_analyzer_tool = RepoAnalyzerTool(repo_path=str(self.repo_path))
        self.repo_metrics_tool = RepositoryMetricsCalculator(repo_path=str(self.repo_path))
        self.pattern_analyzer_tool = PatternAnalyzerTool(repo_path=str(self.repo_path))
        self.grouping_strategy_selector_tool = GroupingStrategySelector(repo_path=str(self.repo_path))
        self.batch_splitter_tool = BatchSplitterTool(repo_path=str(self.repo_path))
        self.group_merging_tool = GroupMergingTool(repo_path=str(self.repo_path))
        self.file_grouper_tool = FileGrouperTool(repo_path=str(self.repo_path))
        self.group_validator_tool = GroupValidatorTool(repo_path=str(self.repo_path))
        self.group_refiner_tool = GroupRefinerTool(repo_path=str(self.repo_path))

        # Create a map for easy access by name if needed elsewhere, though direct access is fine too
        self.tools_map = {
            "repo_analyzer_tool": self.repo_analyzer_tool,
            "repo_metrics_tool": self.repo_metrics_tool,
            "pattern_analyzer_tool": self.pattern_analyzer_tool,
            "grouping_strategy_selector_tool": self.grouping_strategy_selector_tool,
            "batch_splitter_tool": self.batch_splitter_tool,
            "group_merging_tool": self.group_merging_tool,
            "file_grouper_tool": self.file_grouper_tool,
            "group_validator_tool": self.group_validator_tool,
            "group_refiner_tool": self.group_refiner_tool
        }
        logger.info("Tools instantiated.")

    def __getattr__(self, name):
        """
        Dynamically handle requests for tool methods.
        This is called when an attribute is not found through normal means.
        """
        if name in self.tools_map:
            return lambda: self.tools_map[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    @before_kickoff
    def prepare_inputs(self, inputs: Optional[Dict] = None) -> Dict:
        """Prepare inputs before the crew starts."""
        logger.info("Preparing inputs for crew kickoff")
        inputs = inputs or {}
        
        # Always include max_files even if None
        inputs.update({
            'repo_path': str(self.repo_path),
            'max_files': self.max_files,  # This will be None if not provided
            'max_batch_size': self.max_batch_size,
        })
        
        logger.debug(f"Inputs prepared: {inputs}")
        return inputs

    def _save_output_callback(self, step_name: str):
        """Creates a callback function to save task output to JSON."""
        filepath = self.output_dir / f"{step_name}.json"
        
        def save_output(output):
            """Save output to a JSON file and return it unchanged."""
            logger.info(f"Saving output for step '{step_name}' to {filepath}")
            
            try:
                # Clean the output if it's a string and contains code block markers
                if isinstance(output, str):
                    # Remove code block markers
                    output = re.sub(r'```json\s*', '', output)
                    output = re.sub(r'```\s*$', '', output)

                # One-line serialization with a fallback to string conversion
                with open(filepath, 'w') as f:
                    json.dump(output, f, indent=2, default=lambda o: 
                        o.model_dump(mode='json') if hasattr(o, 'model_dump') else str(o))
                    
                logger.info(f"Successfully saved output for '{step_name}'")
            except Exception as e:
                logger.error(f"Failed to save output for '{step_name}': {e}")
                
            return output
        
        return save_output

    # --- Agent Definitions using @agent ---

    @agent
    def pr_manager_agent(self) -> Agent:
        """Creates the PR Manager agent (no tools for hierarchical process)."""
        agent_id = "pr_manager_agent" # Match key in agents.yaml
        logger.info(f"Creating agent: {agent_id}")
        agent_config = self.agents_config[agent_id]
        logger.debug(f"{agent_id} config: {agent_config}")

        return Agent(
            config=self.agents_config[agent_id],
            tools=[],  # Manager agent should not have tools in hierarchical process
            verbose=self.verbose,
            allow_delegation=True  # Manager MUST allow delegation
        )

    @agent
    def analysis_agent(self) -> Agent:
        """Creates the Analysis agent."""
        agent_id = "analysis_agent" # Match key in agents.yaml
        logger.info(f"Creating agent: {agent_id}")
        agent_config = self.agents_config[agent_id]
        logger.debug(f"{agent_id} config: {agent_config}")

        return Agent(
            config=self.agents_config[agent_id],
            tools=[
                self.repo_analyzer_tool,
                self.repo_metrics_tool,
                self.pattern_analyzer_tool,
                self.grouping_strategy_selector_tool,
                self.batch_splitter_tool,
            ],
            verbose=self.verbose,
            allow_delegation=False
        )
    
    @agent
    def batch_processor_agent(self) -> Agent:
        """Creates the Batch Processor agent."""
        agent_id = "batch_processor_agent"
        logger.info(f"Creating agent: {agent_id}")
        return Agent(
            config=self.agents_config[agent_id],
            tools=[
                self.file_grouper_tool,
                self.group_validator_tool,
                self.group_refiner_tool,
            ],
            verbose=self.verbose,
            allow_delegation=False
        )

    @agent
    def merger_refiner_agent(self) -> Agent:
        """Creates the Merger and Refiner agent that handles post-processing tasks."""
        agent_id = "merger_refiner_agent"  # Add this to agents.yaml
        logger.info(f"Creating agent: {agent_id}")
        return Agent(
            config=self.agents_config[agent_id],
            tools=[
                self.group_merging_tool,
                self.group_validator_tool,
                self.group_refiner_tool,
            ],
            verbose=self.verbose,
            allow_delegation=False
        )

    # --- Task Definitions (Using correct pattern) ---
    # Analysis Phase
    @task
    def initial_analysis(self) -> Task:
        task_id = "initial_analysis"
        return Task(
            config=self.tasks_config[task_id],
            agent=self.analysis_agent(), # Assign correct agent
            # Output handled by callback/framework
            callback=self._save_output_callback("step_1_initial_analysis"),
            output_pydantic=RepositoryAnalysis # Specify expected model type
        )

    @task
    def calculate_global_metrics(self) -> Task:
        task_id = "calculate_global_metrics"
        return Task(
            config=self.tasks_config[task_id],
            agent=self.analysis_agent(),
            context=[self.initial_analysis()],
            callback=self._save_output_callback("step_2_global_metrics"),
             output_pydantic=RepositoryMetrics
        )

    @task
    def analyze_global_patterns(self) -> Task:
        task_id = "analyze_global_patterns"
        return Task(
            config=self.tasks_config[task_id],
            agent=self.analysis_agent(),
            context=[self.initial_analysis()],
            callback=self._save_output_callback("step_3_global_patterns"),
             output_pydantic=PatternAnalysisResult
        )

    @task
    def select_grouping_strategy(self) -> Task:
        task_id = "select_grouping_strategy"
        return Task(
            config=self.tasks_config[task_id],
            agent=self.analysis_agent(),
            context=[self.initial_analysis(), self.calculate_global_metrics(), self.analyze_global_patterns()],
            callback=self._save_output_callback("step_4_strategy_decision"),
            output_pydantic=GroupingStrategyDecision
        )

    @task
    def split_into_batches(self) -> Task:
        task_id = "split_into_batches"
        return Task(
            config=self.tasks_config[task_id],
            agent=self.analysis_agent(),
            context=[self.initial_analysis()], # Add others if tool needs them
            callback=self._save_output_callback("step_5_split_batches"),
            output_pydantic=BatchSplitterOutput
            # Note: Ensure max_batch_size from kickoff is available in context if tool needs it
        )
    
    # Manager Orchestration Task with Parallel Processing
    @task
    def coordinate_batch_processing(self) -> Task:
        task_id = "coordinate_batch_processing"
        logger.info(f"Creating manager task with parallel processing: {task_id}")
        return Task(
            config=self.tasks_config[task_id],
            agent=self.pr_manager_agent(), # Manager executes this
            context=[self.split_into_batches(), self.select_grouping_strategy()],
            callback=self._save_output_callback("step_6_coordination_output"),
            async_execution=False  # Disable async execution for parallel processing
            # Output should be List[str] (JSON list of JSON strings)
        )

    # Task to be Delegated (Defined but not in main sequence)
    @task
    def process_single_batch(self) -> Task:
        # This task definition primarily serves to inform the manager about its existence
        # and the agent/tools capable of executing it during delegation.
        task_id = "process_single_batch"
        logger.info(f"Defining delegatable task: {task_id}")
        return Task(
            config=self.tasks_config[task_id],
            agent=self.batch_processor_agent(), # Intended executor
            # No context here, it's provided during delegation by the manager
            # No callback here, result is returned to the manager
            output_pydantic=PRGroupingStrategy # Expecting strategy JSON for the batch
        )

    # Merging & Finalization Phase (Now delegated to merger_refiner_agent)
    @task
    def merge_batch_results(self) -> Task:
        task_id = "merge_batch_results"
        return Task(
            config=self.tasks_config[task_id],
            agent=self.merger_refiner_agent(), # Refiner agent uses the tool now
            context=[self.coordinate_batch_processing(), self.initial_analysis()],
            callback=self._save_output_callback("step_7_merged_groups"),
             output_pydantic=PRGroupingStrategy
        )

    @task
    def final_validation(self) -> Task:
        task_id = "final_validation"
        return Task(
            config=self.tasks_config[task_id],
            agent=self.merger_refiner_agent(), # Refiner agent uses the tool now
            context=[self.merge_batch_results(), self.initial_analysis()],
            callback=self._save_output_callback("step_8_final_validation"),
             output_pydantic=PRValidationResult
        )

    @task
    def final_refinement(self) -> Task:
        task_id = "final_refinement"
        final_output_filename = f"{self.output_dir.name}_final_recommendations"
        return Task(
            config=self.tasks_config[task_id],
            agent=self.merger_refiner_agent(), # Refiner agent uses the tool now
            context=[self.merge_batch_results(), self.final_validation(), self.initial_analysis()],
            callback=self._save_output_callback(final_output_filename),
            output_pydantic=PRGroupingStrategy
        )

    # --- Crew Definition ---
    @crew
    def crew(self) -> Crew:
        """Creates the Hierarchical PR Recommendation crew."""
        logger.info("Assembling the Hierarchical PR Recommendation crew...")
        return Crew(
            agents=[  # Do NOT include manager_agent here
                self.analysis_agent(),
                self.batch_processor_agent(),
                self.merger_refiner_agent(),  # Add the new agent for merging & refining
            ],
            tasks=[  # Define the main sequence controlled by the manager
                self.initial_analysis(),
                self.calculate_global_metrics(),
                self.analyze_global_patterns(),
                self.select_grouping_strategy(),
                self.split_into_batches(),
                self.coordinate_batch_processing(), # Manager orchestrates batches here with parallelism
                self.merge_batch_results(),
                self.final_validation(),
                self.final_refinement(),
                # Note: process_single_batch task is NOT listed here.
                # The manager delegates it dynamically during coordinate_batch_processing
            ],
            process=Process.hierarchical, # Use hierarchical process
            manager_agent=self.pr_manager_agent(), # Our manager agent
            verbose=self.verbose,
            memory=True, # Especially useful for our manager
            # cache=True
        )