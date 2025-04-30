# crew.py
import json
import re
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable

from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, task, crew, before_kickoff
from shared.utils.logging_utils import get_logger
# --- Model Imports ---
from .models.agent_models import (
    RepositoryMetrics, PatternAnalysisResult, GroupingStrategyDecision,
    PRGroupingStrategy, PRValidationResult # PRGroupingStrategy needed for new task output
)
from models.batching_models import BatchSplitterOutput
from shared.models.analysis_models import RepositoryAnalysis

# --- Tool Imports ---
from .tools.repo_analyzer_tool import RepoAnalyzerTool
from .tools.repo_metrics_tool import RepositoryMetricsCalculator
from .tools.pattern_analyzer_tool import PatternAnalyzerTool
from .tools.grouping_strategy_selector_tool import GroupingStrategySelector
from .tools.batch_splitter_tool import BatchSplitterTool
# Import the NEW tool
from .tools.batch_processor_tool import BatchProcessorTool
# Keep Merger/Validator/Refiner for the final agent
from .tools.group_merging_tool import GroupMergingTool
from .tools.group_validator_tool import GroupValidatorTool
from .tools.group_refiner_tool import GroupRefinerTool
# --- NEW: Import PRNamerTool (If we decided to create one, but we decided against it for now) ---
# from .tools.pr_namer_tool import PRNamerTool # Example if we added a tool


logger = get_logger(__name__)

@CrewBase
class SequentialPRCrew:
    """
    Sequential Crew using Process.sequential to analyze repository changes
    and suggest logical PR groupings. Includes LLM-based title refinement.
    """
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self,
                 repo_path: str,
                 max_files: Optional[int] = None,
                 max_batch_size: int = 50, # Used by split_into_batches task
                 verbose: bool | int = False,
                 output_dir: str = 'outputs',
                 manager_llm_name: str = "gpt-4o"):
        """
        Initialize the PR Recommendation crew.

        Args:
            repo_path: Path to the git repository.
            max_files: Maximum number of files to analyze.
            max_batch_size: Target size for batches created by batch_splitter_tool.
            verbose: Verbosity level (True/False or 1/2).
            output_dir: Directory to save intermediate and final outputs.
            manager_llm_name: LLM for potential future manager agent (not used in sequential).
        """
        logger.info(f"Initializing SequentialPRCrew (Consolidated Batch Processor Tool) for repository: {repo_path}")
        logger.info(f"Max files: {max_files}, Max Batch Size: {max_batch_size}, Verbose: {verbose}")

        self.repo_path = Path(repo_path).resolve() # Ensure absolute path
        self.max_files = max_files
        self.max_batch_size = max_batch_size
        self.verbose = verbose
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True) # Ensure output dir exists
        self.manager_llm_name = manager_llm_name

        # Flag for debugging context data between steps
        self.debug_context = True # Set to False to reduce log noise

        # Check repo path
        if not (self.repo_path / ".git").is_dir():
            raise ValueError(f"Not a valid Git repository: {self.repo_path}")

        # --- Tool Instantiation ---
        logger.info("Instantiating tools...")
        # Analysis Agent Tools
        self.repo_analyzer_tool = RepoAnalyzerTool(repo_path=str(self.repo_path))
        self.repo_metrics_tool = RepositoryMetricsCalculator(repo_path=str(self.repo_path))
        self.pattern_analyzer_tool = PatternAnalyzerTool(repo_path=str(self.repo_path))
        self.grouping_strategy_selector_tool = GroupingStrategySelector(repo_path=str(self.repo_path))
        self.batch_splitter_tool = BatchSplitterTool(repo_path=str(self.repo_path))
        # Batch Processor Agent Tool
        self.batch_processor_tool = BatchProcessorTool(repo_path=str(self.repo_path))
        # Merger/Refiner Agent Tools
        self.group_merging_tool = GroupMergingTool(repo_path=str(self.repo_path))
        self.group_validator_tool = GroupValidatorTool(repo_path=str(self.repo_path))
        self.group_refiner_tool = GroupRefinerTool(repo_path=str(self.repo_path))
        # self.pr_namer_tool = PRNamerTool() # Instantiate if we created the tool

        logger.info("Tools instantiated.")

    @before_kickoff
    def prepare_inputs(self, inputs: Optional[Dict] = None) -> Dict:
        """Prepare inputs before the crew starts."""
        logger.info("Preparing inputs for crew kickoff")
        inputs = inputs or {}
        inputs.update({
            'repo_path': str(self.repo_path),
            'max_files': self.max_files if self.max_files is not None else 'None', # Use 'None' string if None
            'max_batch_size': self.max_batch_size,
        })
        logger.debug(f"Inputs prepared: {inputs}")
        return inputs

    # --- _save_output_callback remains the same ---
    def _save_output_callback(self, step_name: str):
        """Creates a callback function to save task output to JSON, cleaning common issues."""
        filepath = self.output_dir / f"{step_name}.json"

        def save_output(output):
            """Save output to a JSON file and return it unchanged for crew context."""
            logger.info(f"Callback invoked for step '{step_name}'. Saving output to {filepath}")
            processed_output = output

            try:
                raw_str = None
                pydantic_obj = None # Store potential pydantic object
                if hasattr(output, 'raw_output') and isinstance(output.raw_output, str):
                    raw_str = output.raw_output
                    processed_output = raw_str # Start with raw if available
                elif isinstance(output, str):
                    raw_str = output
                    processed_output = raw_str
                # --- NEW: Handle Pydantic Object Direct Output ---
                elif hasattr(output, 'model_dump_json'): # Check if it's a Pydantic model
                     pydantic_obj = output
                     processed_output = pydantic_obj.model_dump_json(indent=2) # Use its JSON representation
                     raw_str = processed_output # Treat the JSON string as the raw string now
                     logger.info(f"Output for '{step_name}' is a Pydantic object. Using its JSON representation.")
                # --- END NEW ---

                if raw_str is not None:
                    logger.debug(f"Cleaning raw string output for '{step_name}'")
                    # --- Enhanced Cleaning ---
                    cleaned_str = raw_str.strip()
                    # Remove potential markdown fences even if they have language specifiers
                    cleaned_str = re.sub(r'^```[a-zA-Z]*\s*', '', cleaned_str, flags=re.MULTILINE)
                    cleaned_str = re.sub(r'```\s*$', '', cleaned_str, flags=re.MULTILINE).strip()
                    # Attempt to remove common non-JSON prefixes before the first { or [
                    first_brace = cleaned_str.find('{')
                    first_bracket = cleaned_str.find('[')
                    start_pos = -1
                    if first_brace != -1 and first_bracket != -1:
                        start_pos = min(first_brace, first_bracket)
                    elif first_brace != -1:
                        start_pos = first_brace
                    elif first_bracket != -1:
                        start_pos = first_bracket

                    if start_pos > 0: # If JSON doesn't start at the beginning
                        prefix = cleaned_str[:start_pos].strip()
                        if len(prefix) > 0 and not prefix.endswith(','): # Avoid removing parts of previous JSON structures in lists
                            logger.warning(f"Removing potential non-JSON prefix for '{step_name}': '{prefix[:50]}...'")
                            cleaned_str = cleaned_str[start_pos:]
                    elif start_pos == -1: # Neither { nor [ found
                        logger.warning(f"Cleaned string for '{step_name}' does not appear to be JSON.")
                        # Keep the cleaned string as is, saving logic will handle it
                    # --- End Enhanced Cleaning ---
                    processed_output = cleaned_str
                    logger.debug(f"Cleaned string length: {len(cleaned_str)}")

                # --- Debug Logging (Remains the same) ---
                if self.debug_context:
                   debug_info = f"\n--- DEBUG START: Step '{step_name}' Output ---"
                   debug_info += f"\nOriginal Output Type: {type(output)}"
                   debug_info += f"\nProcessed Output Type: {type(processed_output)}"
                   if pydantic_obj: debug_info += f"\n(Pydantic Object: {type(pydantic_obj)})"
                   if isinstance(processed_output, str):
                       debug_info += f"\nString Length: {len(processed_output)}"
                       debug_info += f"\nString Preview: {processed_output[:200]}..." if len(processed_output) > 200 else processed_output
                   elif isinstance(processed_output, dict): debug_info += f"\nDict Keys: {list(processed_output.keys())}"
                   elif isinstance(processed_output, list): debug_info += f"\nList Length: {len(processed_output)}"
                   debug_info += f"\n--- DEBUG END: Step '{step_name}' Output ---"
                   logger.info(debug_info)
                # --- End Debug Logging ---

                data_to_save = processed_output
                parsed_json_success = False
                if isinstance(processed_output, str):
                    try:
                        # --- Strict JSON Parsing ---
                        # Ensure it's *only* JSON
                        parser = json.JSONDecoder()
                        parsed_json, end_index = parser.raw_decode(processed_output)

                        # Check if there's trailing content after the JSON
                        if end_index < len(processed_output.strip()):
                             trailing_content = processed_output[end_index:].strip()
                             logger.warning(f"Trailing content found after JSON for '{step_name}': '{trailing_content[:50]}...'")
                             # Keep only the parsed JSON part
                             data_to_save = parsed_json
                             logger.info(f"Successfully parsed initial JSON part for '{step_name}'. Discarding trailing content.")
                        else:
                             data_to_save = parsed_json
                             logger.info(f"Successfully parsed cleaned string as JSON for '{step_name}'.")
                        parsed_json_success = True
                        # --- End Strict JSON Parsing ---
                    except json.JSONDecodeError:
                        logger.warning(f"Output for '{step_name}' is string but not valid JSON after cleaning. Saving raw cleaned string.")
                        # data_to_save remains the cleaned string
                elif pydantic_obj:
                     # If we started with a pydantic object, use its validated dictionary form
                     data_to_save = pydantic_obj.model_dump(mode='json')
                     parsed_json_success = True
                     logger.info(f"Using Pydantic object's dictionary representation for saving '{step_name}'.")

                # --- File Saving (Remains the same) ---
                with open(filepath, 'w', encoding='utf-8') as f:
                    if not parsed_json_success and isinstance(data_to_save, str):
                        f.write(data_to_save) # Save the (cleaned) string if JSON parsing failed
                    elif parsed_json_success: # Save the parsed dict/list
                        json.dump(data_to_save, f, indent=2, ensure_ascii=False)
                    else: # Fallback for unexpected types (shouldn't happen often)
                         logger.error(f"Unexpected data type for saving '{step_name}': {type(data_to_save)}. Saving repr.")
                         f.write(repr(data_to_save))

                logger.info(f"Successfully saved output for '{step_name}' to {filepath}")

            except Exception as e:
                logger.error(f"Error during _save_output_callback for '{step_name}': {e}", exc_info=True)
                try: # Attempt to save raw on error
                    error_filepath = self.output_dir / f"{step_name}_ERROR_RAW.txt"
                    with open(error_filepath, 'w', encoding='utf-8') as f_err:
                         f_err.write(f"Error saving original output:\n{e}\n\nRaw Output:\n{repr(output)}") # Use repr for robustness
                    logger.info(f"Saved raw error output to {error_filepath}")
                except Exception as e_raw: logger.error(f"Could not save raw error output: {e_raw}")

            # --- IMPORTANT: Return the *original* output object for CrewAI context ---
            return output # Return original object for CrewAI, not the processed string/dict

        return save_output

    # --- Agent Definitions (remain the same) ---
    @agent
    def analysis_agent(self) -> Agent:
        """Creates the Analysis agent."""
        agent_id = "analysis_agent"
        logger.info(f"Creating agent: {agent_id}")
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
        """Creates the Batch Processor agent (uses the consolidated tool)."""
        agent_id = "batch_processor_agent"
        logger.info(f"Creating agent: {agent_id}")
        return Agent(
            config=self.agents_config[agent_id],
            tools=[
                self.batch_processor_tool,
            ],
            verbose=self.verbose,
            allow_delegation=False
        )

    @agent
    def merger_refiner_agent(self) -> Agent:
        """Creates the Merger and Refiner agent."""
        agent_id = "merger_refiner_agent"
        logger.info(f"Creating agent: {agent_id}")
        return Agent(
            config=self.agents_config[agent_id],
            tools=[
                self.group_merging_tool,
                self.group_validator_tool,
                self.group_refiner_tool,
                # Add PRNamerTool here if we created it
            ],
            verbose=self.verbose,
            allow_delegation=False
        )

    # --- Task Definitions ---

    @task
    def initial_analysis(self) -> Task:
        task_id = "initial_analysis"
        logger.info(f"Defining task: {task_id}")
        return Task(
            config=self.tasks_config[task_id], agent=self.analysis_agent(),
            callback=self._save_output_callback("step_1_initial_analysis"),
            # No output_pydantic needed as description asks for raw JSON string
        )

    @task
    def calculate_global_metrics(self) -> Task:
        task_id = "calculate_global_metrics"
        logger.info(f"Defining task: {task_id}")
        return Task(
            config=self.tasks_config[task_id], agent=self.analysis_agent(),
            context=[self.initial_analysis()],
            callback=self._save_output_callback("step_2_global_metrics"),
            output_pydantic=RepositoryMetrics # Tool returns this
        )

    @task
    def analyze_global_patterns(self) -> Task:
        task_id = "analyze_global_patterns"
        logger.info(f"Defining task: {task_id}")
        return Task(
            config=self.tasks_config[task_id], agent=self.analysis_agent(),
            context=[self.initial_analysis()],
            callback=self._save_output_callback("step_3_global_patterns"),
            output_pydantic=PatternAnalysisResult # Tool returns this
        )

    @task
    def select_grouping_strategy(self) -> Task:
        task_id = "select_grouping_strategy"
        logger.info(f"Defining task: {task_id}")
        return Task(
            config=self.tasks_config[task_id], agent=self.analysis_agent(),
            context=[self.initial_analysis(), self.calculate_global_metrics(), self.analyze_global_patterns()],
            callback=self._save_output_callback("step_4_strategy_decision"),
            output_pydantic=GroupingStrategyDecision # Tool returns this
        )

    @task
    def split_into_batches(self) -> Task:
        task_id = "split_into_batches"
        logger.info(f"Defining task: {task_id}")
        return Task(
            config=self.tasks_config[task_id], agent=self.analysis_agent(),
            context=[self.initial_analysis()],
            callback=self._save_output_callback("step_5_split_batches"),
            # No output_pydantic needed as description asks for raw JSON string
        )

    @task
    def process_batches_and_generate_results(self) -> Task:
        """Task to process all batches using the BatchProcessorTool."""
        task_id = "process_batches_and_generate_results"
        logger.info(f"Defining task: {task_id}")
        return Task(
            config=self.tasks_config[task_id],
            agent=self.batch_processor_agent(),
            context=[
                self.split_into_batches(),
                self.select_grouping_strategy(),
                self.initial_analysis(),
                self.analyze_global_patterns()
            ],
            callback=self._save_output_callback("step_6_processed_batches"),
            # Output is JSON array string, no Pydantic model specified
        )

    @task
    def merge_batch_results(self) -> Task:
        task_id = "merge_batch_results"
        logger.info(f"Defining task: {task_id}")
        return Task(
            config=self.tasks_config[task_id],
            agent=self.merger_refiner_agent(),
            context=[
                self.process_batches_and_generate_results(),
                self.initial_analysis()
            ],
            callback=self._save_output_callback("step_7_merged_groups"),
            output_pydantic=PRGroupingStrategy # Tool returns this Pydantic model as JSON string
        )

    # --- NEW TASK DEFINITION ---
    @task
    def refine_group_names(self) -> Task:
        """Task to refine PR group names using LLM."""
        task_id = "refine_group_names"
        logger.info(f"Defining task: {task_id}")
        return Task(
            config=self.tasks_config[task_id],
            agent=self.merger_refiner_agent(), # Reuse agent
            context=[self.merge_batch_results()], # Needs merged groups
            # No specific tool needed if agent handles LLM logic based on description
            callback=self._save_output_callback("step_7b_refined_names"), # New step name
            output_pydantic=PRGroupingStrategy # Expects the updated strategy object as JSON string
        )
    # --- END NEW TASK ---

    @task
    def final_validation(self) -> Task:
        task_id = "final_validation"
        logger.info(f"Defining task: {task_id}")
        return Task(
            config=self.tasks_config[task_id],
            agent=self.merger_refiner_agent(),
            context=[self.refine_group_names()], # Context updated to use name-refined groups
            callback=self._save_output_callback("step_8_final_validation"),
            output_pydantic=PRValidationResult # Tool returns this
        )

    @task
    def final_refinement(self) -> Task:
        task_id = "final_refinement"
        logger.info(f"Defining task: {task_id}")
        final_output_filename = f"{self.output_dir.name}_final_recommendations"
        return Task(
            config=self.tasks_config[task_id],
            agent=self.merger_refiner_agent(),
            context=[
                self.refine_group_names(), # Use name-refined groups
                self.final_validation(),
                self.initial_analysis()
            ],
            callback=self._save_output_callback(final_output_filename),
            output_pydantic=PRGroupingStrategy # Tool returns this
        )

    # --- Crew Definition ---
    @crew
    def crew(self) -> Crew:
        """Creates the Sequential PR Recommendation crew with LLM title refinement."""
        logger.info("Assembling the Sequential PR Recommendation crew (with title refinement)...")

        # --- UPDATED TASK SEQUENCE ---
        all_tasks = [
            self.initial_analysis(),
            self.calculate_global_metrics(),
            self.analyze_global_patterns(),
            self.select_grouping_strategy(),
            self.split_into_batches(),
            self.process_batches_and_generate_results(),
            self.merge_batch_results(),
            self.refine_group_names(), # Insert the new naming task here
            self.final_validation(),
            self.final_refinement(),
        ]
        # --- END UPDATED SEQUENCE ---

        logger.info(f"Total tasks in sequence: {len(all_tasks)}")
        if self.verbose > 1:
            for idx, t in enumerate(all_tasks):
                 # Use task_id from config if available, fallback to method name
                 task_name = t.config.get('name', t.name) if hasattr(t, 'config') and t.config else t.name
                 desc = getattr(t, 'description', 'N/A')
                 logger.debug(f"Task {idx} ({task_name}): {desc[:70]}...")


        return Crew(
            agents=[
                self.analysis_agent(),
                self.batch_processor_agent(),
                self.merger_refiner_agent(),
            ],
            tasks=all_tasks, # Use the updated list
            process=Process.sequential,
            verbose=self.verbose,
            memory=True, # Keep memory enabled
        )