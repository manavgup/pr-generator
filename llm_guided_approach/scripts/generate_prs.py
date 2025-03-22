#!/usr/bin/env python3
"""
Main script for generating PR suggestions using the LLM-guided approach.
"""
import os
import sys
import json
import logging
import argparse
from datetime import datetime
from shared.utils.logging_utils import configure_logging

# Add current directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from llm_guided_approach
from llm_guided_approach import PRGenerator
from shared.config.llm_config import LLMProvider, create_llm_config

# Ensure output directory exists
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
# Configure logging
logger = logging.getLogger(__name__)

def display_pr_suggestions(pr_suggestions):
    """
    Display PR suggestions in a formatted way.
    
    Args:
        pr_suggestions: Dictionary with PR suggestions
    """
    if "error" in pr_suggestions:
        print(f"\nError generating PR suggestions: {pr_suggestions['error']}")
        return
    
    suggestions = pr_suggestions.get("pr_suggestions", [])
    if not suggestions:
        print("\nNo PR suggestions were generated.")
        return
    
    for i, pr in enumerate(suggestions, 1):
        print(f"\n{'='*80}")
        print(f"PR #{i}: {pr['title']}")
        print(f"{'='*80}")
        
        print("\nDescription:")
        print("-" * 40)
        # Split description into paragraphs for better readability
        if pr.get('description'):
            for para in pr['description'].split('\n'):
                print(para)
        else:
            print("No description provided.")
        
        print("\nFiles:")
        print("-" * 40)
        # Group files by directory for better organization
        files_by_dir = {}
        for file in pr.get('files', []):
            dir_name = os.path.dirname(file) or '(root)'
            if dir_name not in files_by_dir:
                files_by_dir[dir_name] = []
            files_by_dir[dir_name].append(os.path.basename(file))
        
        # Print files organized by directory
        for dir_name, files in sorted(files_by_dir.items()):
            print(f"\n  {dir_name}/")
            for file in sorted(files):
                print(f"    - {file}")
        
        print(f"{'='*80}")

def save_pr_suggestions(pr_suggestions, output_path):
    """
    Save PR suggestions to a file.
    
    Args:
        pr_suggestions: Dictionary with PR suggestions
        output_path: Path to save PR suggestions
    """
    try:
        with open(output_path, 'w') as f:
            json.dump(pr_suggestions, f, indent=2)
        logger.info(f"Saved PR suggestions to {output_path}")
        print(f"\nSaved PR suggestions to {output_path}")
    except Exception as e:
        logger.error(f"Error saving PR suggestions: {e}")
        print(f"\nError saving PR suggestions: {e}")

def get_default_output_filename(repo_path, provider, model):
    """
    Generate a default output filename based on repo, provider and model.
    
    Args:
        repo_path: Path to the repository
        provider: LLM provider name
        model: Model name
        
    Returns:
        Default output filename
    """
    # Extract repo name from path
    repo_name = os.path.basename(os.path.normpath(repo_path))
    # Get timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate filename
    model_name = model or ("llama3" if provider == "ollama" else "gpt-4o-mini")
    filename = f"pr_suggestions_{repo_name}_{provider}_{model_name}_{timestamp}.json"
    
    # Return full path
    return os.path.join(OUTPUT_DIR, filename)

def main():
    """Main function for the PR generator script."""
    parser = argparse.ArgumentParser(description='Generate PR suggestions for a git repository')
    parser.add_argument('repo_path', help='Path to the git repository')
    parser.add_argument('--provider', default='openai', choices=['ollama', 'openai', 'anthropic'], 
                       help='LLM provider to use (ollama, openai, or anthropic)')
    parser.add_argument('--max-files', type=int, default=20, help='Maximum number of files per PR')
    parser.add_argument('--base-url', default='http://localhost:11434', 
                       help='Base URL for the LLM provider (required for Ollama, optional for others)')
    parser.add_argument('--model', help='Model to use for analysis (defaults: llama3 for ollama, gpt-4o-mini for openai, claude-3-haiku for anthropic)')
    parser.add_argument('--api-key', help='API key for the provider (required for OpenAI and Anthropic)')
    parser.add_argument('--output', help='Path to save PR suggestions (optional, defaults to output directory)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--use-embeddings', action='store_true', help='Use embeddings for file grouping')
    parser.add_argument("--log-file", default=None, 
                   help="Path to log file (defaults to timestamped file in output directory)")
    
    args = parser.parse_args()
    
    if args.log_file is None:
        args.log_file = os.path.join(OUTPUT_DIR, f'pr_generator_llm_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')


    if args.verbose:
        configure_logging(
        log_file=args.log_file,
        verbose=args.verbose
    )
        
    logger.debug("DEBUG: This is a verbose log test")

    # Set default output path if not provided
    if not args.output:
        args.output = get_default_output_filename(args.repo_path, args.provider, args.model)
    # If output path doesn't include directory, put it in the output directory
    elif not os.path.dirname(args.output):
        args.output = os.path.join(OUTPUT_DIR, args.output)
    
    logger.info(f"Starting PR generator for repository: {args.repo_path}")
    logger.info(f"Using provider: {args.provider}, model: {args.model or 'default'}")
    logger.info(f"Output will be saved to: {args.output}")
    
    try:
        # Create PR generator with the new approach
        generator = PRGenerator(
            repo_path=args.repo_path,
            max_files_per_pr=args.max_files,
            llm_provider=args.provider,
            llm_model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            verbose=args.verbose,
        )
        
        # Generate PR suggestions
        print("\nGenerating PR suggestions...")
        pr_suggestions = generator.create_prs()
        
        # Display PR suggestions
        display_pr_suggestions(pr_suggestions)
        
        # Save PR suggestions
        save_pr_suggestions(pr_suggestions, args.output)
    
    except Exception as e:
        logger.exception(f"Error in main: {e}")
        print(f"Error: {e}")
    
    logger.info("PR generator completed")

if __name__ == "__main__":
    main()