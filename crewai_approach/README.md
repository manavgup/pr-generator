# PR Generator - CrewAI Approach

A specialized tool for analyzing Git repository changes and automatically generating logical pull request groupings using a multi-agent system powered by CrewAI.

## Overview

The CrewAI approach uses multiple specialized AI agents working together to:

1. **Analyze code changes** - Extract and understand repository modifications
2. **Strategize PR groups** - Organize changes into logical, reviewable units
3. **Generate PR content** - Create descriptive titles, branch names, and documentation
4. **Validate suggestions** - Ensure all changed files are included and properly balanced

## Architecture

This implementation is built on [CrewAI](https://github.com/joaomdmoura/crewAI), which enables orchestration of multiple specialized agents:

- **Code Analyzer Agent**: Examines git diffs to understand technical implications
- **PR Strategist Agent**: Groups changes into logical pull requests
- **Content Generator Agent**: Creates descriptive titles and documentation
- **Validator Agent**: Ensures completeness and balance of suggestions

## Usage

### Prerequisites

- Python 3.10+
- Git repository with changes (unstaged/uncommitted)
- LLM access (OpenAI API key or local Ollama instance)

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Running the PR Generator

```bash
# Using OpenAI
python main.py /path/to/repository --provider openai --api-key YOUR_API_KEY

# Using Ollama (local)
python main.py /path/to/repository --provider ollama --model llama3
```

### Command Line Options

```
usage: main.py [-h] [--provider {ollama,openai}] [--model MODEL] [--llm-url LLM_URL] [--api-key API_KEY] [--output OUTPUT] [--verbose] [--dry-run] [--check-files] repo_path

Generate PR suggestions for a git repository using CrewAI

positional arguments:
  repo_path             Path to the git repository

options:
  -h, --help            show this help message and exit
  --provider {ollama,openai}
                        LLM provider to use (ollama or openai)
  --model MODEL         Model to use for analysis (defaults: llama3 for ollama, gpt-4o-mini for openai)
  --llm-url LLM_URL     URL for the Ollama service (only used with ollama provider)
  --api-key API_KEY     API key for OpenAI (only used with openai provider)
  --output OUTPUT       Path to save PR suggestions (optional)
  --verbose             Enable verbose logging
  --dry-run             Do not create actual PRs
  --check-files         Check if files are readable before running
```

## Project Structure

```
crewai_approach/
├── pr_generator/
│   ├── src/
│   │   └── pr_generator/
│   │       ├── config/
│   │       │   ├── agents.yaml     # Agent configuration
│   │       │   └── tasks.yaml      # Task definitions
│   │       ├── tools/
│   │       │   ├── git_tools.py    # Git repository analysis
│   │       │   ├── grouping_tool.py # File grouping logic
│   │       │   └── validation_tools.py # Validation utilities
│   │       ├── crew.py             # CrewAI setup and orchestration
│   │       └── main.py             # CLI entry point
│   └── tests/                      # Test suite
└── run_crew_pr.py                  # Alternative entry point
```

## How It Works

1. **Git Analysis**: The system analyzes the repository to identify changed files and their details
2. **File Grouping**: Changed files are organized into logical groups based on:
   - Directory structure
   - File relationships
   - Change patterns
   - Technical dependencies
3. **Content Generation**: Each PR group gets meaningful titles and descriptions
4. **Validation**: System verifies that all changed files are included in the suggestions

## Customization

The system can be customized by modifying:

- `config/agents.yaml`: Change agent roles, goals and backstories
- `config/tasks.yaml`: Adjust the expected outputs and task descriptions
- LLM settings: Change models or providers via command line arguments

## Development Status

This is a work in progress. Core functionality works but the end-to-end flow is still being refined.

## Requirements

- Python 3.10+
- crewai
- pydantic
- PyYAML
- GitPython (optional, for advanced features)