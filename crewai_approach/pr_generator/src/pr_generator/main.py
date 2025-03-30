#!/usr/bin/env python
import sys
import os
import json
import logging
import argparse
from datetime import datetime

from typing import List, Dict, Any, Optional

from .crew import PRGenerator
from shared.config.llm_config import LLMConfig, LLMProvider
from shared.models.pr_models import PRSuggestion
from shared.utils.logging_utils import configure_logging

# Create logs directory if it doesn't exist
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging with enhanced setup
log_file = os.path.join(LOG_DIR, f"pr_generator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logger = configure_logging(log_file=log_file)

def display_pr_suggestions(pr_suggestion: PRSuggestion) -> None:
    """
    Display PR suggestions in a formatted way.
    
    Args:
        pr_suggestion: PRSuggestion object containing PR suggestions
    """
    suggestions = pr_suggestion.pr_suggestions
    
    if not suggestions:
        print("\nNo PR suggestions were generated.")
        return
    
    for i, pr in enumerate(suggestions, 1):
        print(f"\n{'='*80}")
        print(f"PR #{i}: {pr.title}")
        print(f"{'='*80}")
        
        print("\nDescription:")
        print("-" * 40)
        # Split description into paragraphs for better readability
        if pr.description:
            for para in pr.description.split('\n'):
                print(para)
        else:
            print("No description provided.")
        
        print("\nRationale:")
        print("-" * 40)
        print(pr.rationale)
        
        print("\nFiles:")
        print("-" * 40)
        # Group files by directory for better organization
        files_by_dir = {}
        for file in pr.files:
            dir_name = os.path.dirname(file) or '(root)'
            if dir_name not in files_by_dir:
                files_by_dir[dir_name] = []
            files_by_dir[dir_name].append(os.path.basename(file))
        
        # Print files organized by directory
        for dir_name, files in sorted(files_by_dir.items()):
            print(f"\n  {dir_name}/")
            for file in sorted(files):
                print(f"    - {file}")
        
        print(f"\nSuggested Branch: {pr.suggested_branch}")
        print(f"{'='*80}")
    
    # Print additional information from PRSuggestion if available
    if pr_suggestion.description:
        print(f"\nOverall Description: {pr_suggestion.description}")
    if pr_suggestion.message:
        print(f"\nMessage: {pr_suggestion.message}")
    if pr_suggestion.total_groups:
        print(f"\nTotal Groups: {pr_suggestion.total_groups}")

def save_pr_suggestions(pr_suggestion: PRSuggestion, output_path: Optional[str] = None) -> str:
    """
    Save PR suggestions to a file.
    
    Args:
        pr_suggestion: PRSuggestion object
        output_path: Path to save PR suggestions (optional)
        
    Returns:
        Path where the suggestions were saved
    """
    try:
        # Use the provided path or create a default one
        if not output_path:
            output_path = f"pr_suggestions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # Convert to dictionary for JSON serialization
        data = pr_suggestion.model_dump()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved PR suggestions to {output_path}")
        print(f"\nSaved PR suggestions to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error saving PR suggestions: {e}")
        print(f"\nError saving PR suggestions: {e}")
        # Try an alternative path if the original fails
        alt_path = f"fallback_pr_suggestions_{int(datetime.now().timestamp())}.json"
        try:
            with open(alt_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved PR suggestions to alternative path: {alt_path}")
            print(f"\nSaved PR suggestions to alternative path: {alt_path}")
            return alt_path
        except Exception as alt_e:
            logger.error(f"Error saving to alternative path: {alt_e}")
            print(f"\nError saving to alternative path: {alt_e}")
            return ""

# In main.py, modify the display_validation_results function:
def display_validation_results(validation_result: Dict[str, Any]) -> None:
    """Display validation results."""
    if not validation_result:
        return
        
    print(f"\n{'='*80}")
    print("VALIDATION RESULTS")
    print(f"{'='*80}")
    print(f"Status: {validation_result.get('validation_status', 'Unknown')}")
    print(f"Files: {validation_result.get('files_in_prs', 0)}/{validation_result.get('total_changed_files', 0)} included in PRs")
    
    missing_files = validation_result.get('missing_files', [])
    if missing_files:
        print(f"\nMissing Files: {len(missing_files)}")
        for file in missing_files[:10]:
            print(f"  - {file}")
        if len(missing_files) > 10:
            print(f"  - ... and {len(missing_files) - 10} more")
    
    suggestions = validation_result.get('suggestions', [])
    if suggestions:
        print("\nSuggestions:")
        for suggestion in suggestions:
            # Check if suggestion is a dictionary or string
            if isinstance(suggestion, dict):
                print(f"  - {suggestion.get('message', '')}")
            else:
                print(f"  - {suggestion}")

def main():
    """Main function for the PR generator script using CrewAI."""
    global logger
    
    parser = argparse.ArgumentParser(description='Generate PR suggestions for a git repository using CrewAI')
    parser.add_argument('repo_path', help='Path to the git repository')
    parser.add_argument('--provider', default='ollama', choices=['ollama', 'openai'], 
                       help='LLM provider to use (ollama or openai)')
    parser.add_argument('--model', help='Model to use for analysis (defaults: llama3 for ollama, gpt-4o-mini for openai)')
    parser.add_argument('--llm-url', default='http://localhost:11434', 
                       help='URL for the Ollama service (only used with ollama provider)')
    parser.add_argument('--api-key', help='API key for OpenAI (only used with openai provider)')
    parser.add_argument('--output', help='Path to save PR suggestions (optional)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--dry-run', action='store_true', default=True, help='Do not create actual PRs')
    parser.add_argument('--check-files', action='store_true', help='Check if files are readable before running')
    
    args = parser.parse_args()
    # Reconfigure logging with the command-line verbose setting
    if args.verbose:
        logger = configure_logging(log_file=log_file, verbose=True)
        logger.info("Verbose logging enabled from command line")
    
    # Ensure repo_path is absolute
    repo_path = os.path.abspath(args.repo_path)
    
    # Check if the repo path exists
    if not os.path.exists(repo_path):
        logger.error(f"Repository path does not exist: {repo_path}")
        print(f"Error: Repository path does not exist: {repo_path}")
        return 1
        
    # Check if it's a git repository
    if not os.path.exists(os.path.join(repo_path, '.git')):
        logger.error(f"Not a git repository: {repo_path}")
        print(f"Error: Not a git repository: {repo_path}")
        return 1
    
    logger.info(f"Starting PR generator with CrewAI for repository: {repo_path}")
    
    # Create tool_outputs directory if it doesn't exist
    tool_outputs_dir = "tool_outputs"
    os.makedirs(tool_outputs_dir, exist_ok=True)
    
    # If check-files is enabled, check if we can read files
    if args.check_files:
        try:
            from shared.git_operations import GitOperations
            git_ops = GitOperations(repo_path)
            changes = git_ops.get_changed_files()
            logger.info(f"Successfully read {len(changes)} changed files")
            print(f"Successfully read {len(changes)} changed files")
        except Exception as e:
            logger.error(f"Error reading files: {e}")
            print(f"Error reading files: {e}")
            return 1
    
    try:
        # Create LLM configuration
        provider = LLMProvider.OPENAI if args.provider.lower() == "openai" else LLMProvider.OLLAMA
        
        # Determine model based on provider
        if not args.model:
            model = "gpt-4o-mini" if provider == LLMProvider.OPENAI else "llama3"
        else:
            model = args.model
        
        # Get API key from environment if not provided
        api_key = args.api_key
        if provider == LLMProvider.OPENAI and not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                logger.error("OpenAI API key not provided and not found in environment")
                print("Error: OpenAI API key not provided and not found in environment")
                return 1
        
        # Create LLM configuration
        llm_config = LLMConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=args.llm_url if provider == LLMProvider.OLLAMA else None,
            temperature=0.7
        )
        
        # Log LLM details
        logger.info(f"Using LLM provider: {provider.value}, model: {model}")
        if provider == LLMProvider.OLLAMA:
            logger.info(f"Ollama URL: {args.llm_url}")
        
        # Create PR generator crew
        logger.info(f"Creating PR Generator")
        pr_generator = PRGenerator(
            repo_path=repo_path,
            llm_config=llm_config,
            dry_run=args.dry_run,
            verbose=args.verbose
        )
        
        # Generate PR suggestions
        print("\nGenerating PR suggestions with CrewAI...")
        logger.info("Starting CrewAI PR generation process")
        
        # Get crew instance and log details
        crew_instance = pr_generator.crew()
        logger.info(f"Initialized crew with {len(crew_instance.agents)} agents and {len(crew_instance.tasks)} tasks")
        
        # Kick off the crew and get results
        logger.info("Executing CrewAI process")
        result = crew_instance.kickoff()
        logger.info("CrewAI process completed")
        
        # Process and display results
        if hasattr(result, 'pydantic') and result.pydantic:
            logger.info(f"Successfully generated PR suggestions")
            # Save results
            output_path = args.output or f"pr_suggestions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            save_pr_suggestions(result.pydantic, output_path)
            # Display results
            display_pr_suggestions(result.pydantic)
            if hasattr(result.pydantic, 'validation_result') and result.pydantic.validation_result:
                display_validation_results(result.pydantic.validation_result)
        else:
            logger.warning("No structured PR suggestions were generated")
            print("\nNo structured PR suggestions were generated.")
            # Try to save raw output
            if hasattr(result, 'raw') and result.raw:
                logger.info("Saving raw output")
                raw_output_path = f"raw_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(raw_output_path, 'w', encoding='utf-8') as f:
                    f.write(result.raw)
                print(f"Saved raw output to: {raw_output_path}")
        
        logger.info("PR generation process completed successfully")
        return 0
        
    except Exception as e:
        logger.exception(f"Error in main: {e}")
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())