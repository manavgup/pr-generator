import json
import time
import subprocess
from crewai import Agent, Crew, Process, Task
from pydantic import BaseModel
from crewai_approach.pr_generator.src.pr_generator.tools.git_tools import (
    GitAnalysisTool, 
    QuickGitAnalysisTool, 
    GitAnalysisOutput
)

# Utility function to directly verify file count
def verify_git_file_count(repo_path):
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only"],
            capture_output=True,
            text=True,
            cwd=repo_path
        )
        files = [f for f in result.stdout.splitlines() if f.strip()]
        return len(files)
    except Exception as e:
        print(f"Error verifying file count: {e}")
        return 0

# Configuration
REPO_PATH = '/Users/mg/mg-work/manav/work/ai-experiments/rag_modulo'

# First check the actual file count using a direct git command
print(f"\nDirect git command check on {REPO_PATH}:")
direct_count = verify_git_file_count(REPO_PATH)
print(f"Number of changed files (direct git command): {direct_count}")

# Set up the GitHub change agent
print("\nSetting up the GitHub change agent...")
github_change_agent = Agent(
    role="Github repo changes analyzer agent",
    goal="Analyze code changes to understand patterns, relationships, and technical implications",
    backstory="""You are an expert code reviewer with deep understanding of software architecture and design patterns.""",
    verbose=False,
    allow_delegation=False,
    llm="gpt-4o",
)

# Initialize both tools
print("\nInitializing tools...")
tool_init_start = time.time()
git_analysis_tool = GitAnalysisTool(REPO_PATH)
quick_git_analysis_tool = QuickGitAnalysisTool(REPO_PATH)
tool_init_end = time.time()
print(f"Tool initialization time: {tool_init_end - tool_init_start:.2f} seconds")

# Create task with both tools
print("\nCreating task with both tools...")
git_changes_task = Task(
    description="""Analyze the Git changes in the repository, and identify patterns, relationships, and technical implications of the changes.
    Use the quick_git_analysis tool for faster results when you don't need full diffs, and the analyze_git_changes tool when you need detailed diffs.""",
    expected_output="A JSON object with detailed analysis of all changes.",
    agent=github_change_agent,
    tools=[git_analysis_tool, quick_git_analysis_tool],
    output_pydantic=GitAnalysisOutput,
)

# Instantiate crew with both tools
print("\nCreating crew...")
crew = Crew(
    agents=[github_change_agent],
    tasks=[git_changes_task],
    verbose=True,
    process=Process.sequential,
)

# Run the crew
print("\nStarting crew execution...")
crew_start = time.time()
result = crew.kickoff()
crew_end = time.time()
print(f"Crew execution time: {crew_end - crew_start:.2f} seconds")

print("\n=============================================")
print("Detailed timing breakdown:")
print(f"- Tool initialization: {tool_init_end - tool_init_start:.2f} seconds")
print(f"- Crew execution (including API calls): {crew_end - crew_start:.2f} seconds")
print("=============================================")

# Access results
print("\nAccessing Properties")
total_files_changed = result.pydantic.total_files_changed
directory_summaries = result.pydantic.directory_summaries
print("total_files_changed:", total_files_changed)
print("directory_summaries count:", len(directory_summaries))
print("Directory names:", [d.name for d in directory_summaries])  # Fixed to use .name instead of get()

# Verify file counts
print("\nVerifying file counts:")
print(f"Direct git command: {direct_count} files")
print(f"Tool reported: {total_files_changed} files")

# Print full result structure for debugging
print("\nFull Result Structure:")
print("Result type:", type(result))
print("Available attributes:", dir(result))