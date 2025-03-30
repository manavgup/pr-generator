from crewai import Agent, Crew, Task, Process, LLM
from crewai.project import CrewBase, agent, task, crew, before_kickoff
import logging

from .tools.git_tools import GitAnalysisTool
from .tools.grouping_tool import GroupingTool
from shared.models.pr_models import ChangeAnalysis, PullRequestGroup, PRSuggestion
from .tools.validation_tools import ValidationTool, PRRebalancer
from .tools.pr_strategy_tools import (
    SummarizeChangesTool,
    GetDirectoryDetailsTool,
    CreatePRGroupsTool
)
import json

# Configure logger
logger = logging.getLogger(__name__)

@CrewBase
class PRGenerator:
    """PR Generator crew that analyzes code changes and suggests pull requests"""

    # Paths to YAML configuration files
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self, repo_path, llm_config=None, dry_run=True, verbose=True):
        """
        Initialize the PR Generator crew.
        
        Args:
            repo_path: Path to the git repository
            llm_config: LLM configuration (provider, model, etc.)
            dry_run: If True, don't create actual PRs
            verbose: If True, enable verbose logging
        """
        logger.info(f"Initializing PRGenerator for repository: {repo_path}")
        logger.info(f"LLM Config: {llm_config}")
        logger.info(f"Dry Run: {dry_run}, Verbose: {verbose}")

        self.repo_path = repo_path
        self.llm_config = llm_config
        self.dry_run = dry_run
        self.verbose = verbose
        
        # Set up tools
        logger.info("Setting up tools for PR generation")
        self.git_analysis_tool = GitAnalysisTool(repo_path)
        self.grouping_tool = GroupingTool()

        # Set up validation tools
        self.validation_tool = ValidationTool()
        self.rebalancer_tool = PRRebalancer()

        # PR strategy tools
        self.summarize_changes_tool = SummarizeChangesTool()
        self.get_directory_details_tool = GetDirectoryDetailsTool()
        self.create_pr_groups_tool = CreatePRGroupsTool()
        

    @before_kickoff
    def prepare_inputs(self, inputs):
        """Prepare inputs before the crew starts"""
        logger.info("Preparing inputs for crew kickoff")
        inputs['repo_path'] = self.repo_path
        logger.debug(f"Inputs prepared: {inputs}")
        return inputs

    @agent
    def code_analyzer(self) -> Agent:
        """Create the Code Analyzer agent"""
        logger.info("Creating Code Analyzer agent")
        
        # Load agent configuration
        agent_config = self.agents_config['code_analyzer']
        logger.debug(f"Code Analyzer agent config: {agent_config}")
        
        # Prepare LLM kwargs if provided
        llm_kwargs = self._configure_agent_llm(agent_config)
        
        # Create and return agent
        agent = Agent(
            config=agent_config,
            tools=[self.git_analysis_tool],
            verbose=self.verbose,
            **llm_kwargs
        )
        logger.info("Code Analyzer agent created successfully")
        return agent

    @agent
    def pr_strategist_agent(self) -> Agent:
        """Create the PR Strategist agent"""
        logger.info("Creating PR Strategist agent")
        
        # Load agent configuration
        agent_config = self.agents_config['pr_strategist']
        logger.debug(f"PR Strategist agent config: {agent_config}")
        
        # Prepare LLM kwargs if provided
        llm_kwargs = self._configure_agent_llm(agent_config)
        
        # Create and return agent
        agent = Agent(
            config=agent_config,
            tools=[
                self.summarize_changes_tool,
                self.get_directory_details_tool,
                self.create_pr_groups_tool
            ],
            verbose=self.verbose,
            **llm_kwargs
        )
        logger.info("PR Strategist agent created successfully")
        return agent

    @agent
    def pr_content_generator(self) -> Agent:
        """Create the PR Content Generator agent"""
        logger.info("Creating PR Content Generator agent")
        
        # Load agent configuration
        agent_config = self.agents_config['pr_content_generator']
        logger.debug(f"PR Content Generator agent config: {agent_config}")
        
        # Prepare LLM kwargs if provided
        llm_kwargs = self._configure_agent_llm(agent_config)
        
        # Create and return agent with the new strategy tools
        agent = Agent(
            config=agent_config,
            tools=[
                self.summarize_changes_tool,
                self.get_directory_details_tool,
                self.create_pr_groups_tool
            ],
            verbose=self.verbose,
            **llm_kwargs
        )
        logger.info("PR Content Generator agent created successfully")
        return agent

    @task
    def analyze_code_task(self) -> Task:
        """Create the code analysis task"""
        logger.info("Creating code analysis task")
        
        # Load task configuration
        task_config = self.tasks_config['analyze_code_task']
        logger.debug(f"Code Analysis task config: {task_config}")

        def save_output(output):
            try:
                with open('code_analyzer_output.json', 'w') as f:
                    json.dump(output, f, indent=2)
                logger.info("Saved code analyzer output to file")
            except Exception as e:
                logger.error(f"Failed to save output: {e}")
            return output
        
        # Create and return task
        task = Task(config=task_config,
                    output_pydantic=ChangeAnalysis,
                    callback=save_output)
        logger.info("Code Analysis task created successfully")
        return task

    @task
    def strategy_task(self) -> Task:
        """Create the PR strategy task"""
        logger.info("Creating PR strategy task")
        
        # Load task configuration
        task_config = self.tasks_config['strategy_task']
        logger.info(f"PR Strategy task config: {task_config}")
        
        # Create and return task
        task = Task(config=task_config,
                    output_pydantic=PullRequestGroup,
                    context=[self.analyze_code_task()])
        logger.info(f"PR Strategy task created successfully: {task}")
        return task

    @task
    def content_task(self) -> Task:
        """Create the PR content task"""
        logger.info("Creating PR content task")
        
        # Load task configuration
        task_config = self.tasks_config['content_task']
        logger.debug(f"PR Content task config: {task_config}")
        
        # Create and return task
        task = Task(config=task_config,
                    output_pydantic=PRSuggestion)
        logger.info("PR Content task created successfully")
        return task
    
    @agent
    def pr_validator(self) -> Agent:
        """Create the PR Validator agent"""
        logger.info("Creating PR Validator agent")
        
        # Load agent configuration
        agent_config = self.agents_config['pr_validator']
        logger.debug(f"PR Validator agent config: {agent_config}")
        
        # Prepare LLM kwargs if provided
        llm_kwargs = self._configure_agent_llm(agent_config)
        
        # Create and return agent
        agent = Agent(
            config=agent_config,
            tools=[self.validation_tool, self.rebalancer_tool],
            verbose=self.verbose,
            **llm_kwargs
        )
        logger.info("PR Validator agent created successfully")
        return agent

    @task
    def validation_task(self) -> Task:
        """Create the PR validation task"""
        logger.info("Creating PR validation task")
        
        # Load task configuration
        task_config = self.tasks_config['validation_task']
        logger.debug(f"PR Validation task config: {task_config}")
        
        # Create and return task
        task = Task(
            config=task_config,
            output_pydantic=PRSuggestion,
            context=[self.analyze_code_task(), self.content_task()]
        )
        logger.info("PR Validation task created successfully")
        return task

    @crew
    def crew(self) -> Crew:
        """Creates the PR Generator crew"""
        logger.info("Creating PR Generator crew")
        
        # Create crew
        crew = Crew(
            agents=self.agents,   # Automatically collected by @agent decorator
            tasks=self.tasks,     # Automatically collected by @task decorator
            process=Process.sequential,
            verbose=True
        )
        
        logger.info("PR Generator crew created successfully")
        logger.info(f"Crew agents: {self.agents}")
        logger.info(f"Crew tasks: {self.tasks}")
        
        return crew
    
    def _get_llm_kwargs(self):
        """Get LLM configuration kwargs based on provider"""
        if not self.llm_config:
            logger.warning("No LLM configuration provided")
            return {}
        
        # Log LLM provider details
        logger.info(f"Configuring LLM with provider: {self.llm_config.provider}")
        
        # Check provider
        if self.llm_config.provider.lower() == 'openai':
            logger.debug("Configuring OpenAI LLM")
            return {
                'openai_api_key': self.llm_config.api_key,
                'model': self.llm_config.model,
                'temperature': self.llm_config.temperature
            }
        else:  # ollama
            logger.debug("Configuring Ollama LLM")
            # For Ollama, we need to use the model_name format with 'ollama/' prefix
            # and put the base_url in api_base
            model_name = f"ollama/{self.llm_config.model}"
            return {
                'model': model_name,
                'api_base': self.llm_config.base_url or "http://localhost:11434/api",
                'temperature': self.llm_config.temperature
            }
    
    def _configure_agent_llm(self, agent_config):
        """Configure the LLM for an agent."""
        logger.info(f"Configuring LLM for {agent_config.get('role', 'agent')}")
        
        if not self.llm_config:
            logger.warning("No LLM configuration provided")
            return {}
        
        # Check provider
        if self.llm_config.provider.lower() == 'openai':
            logger.debug("Configuring OpenAI LLM")
            return {
                'openai_api_key': self.llm_config.api_key,
                'model': self.llm_config.model,
                'temperature': self.llm_config.temperature
            }
        else:  # ollama
            logger.debug("Creating dedicated Ollama LLM")
            ollama_model_name = f"ollama/{self.llm_config.model}"
            logger.info(f"Using Ollama model: {ollama_model_name}")
            # Create a dedicated Ollama LLM instance
            ollama_llm = LLM(
                model=f"ollama/{self.llm_config.model}",
                base_url=self.llm_config.base_url or "http://localhost:11434"
            )
            return {'llm': ollama_llm}  # Return the LLM directly