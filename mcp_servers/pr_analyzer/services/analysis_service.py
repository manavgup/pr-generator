"""
Analysis service for MCP PR Analyzer
Modified to work with existing crewai_approach imports
"""
import asyncio
import sys
import json
import logging
import asyncio
import tempfile
import os
import shutil # Import shutil for temporary directory cleanup
from pathlib import Path
from typing import Dict, Any, Optional, List

from mcp.types import TextContent # Import TextContent

# Add the project root to sys.path so crewai_approach imports work
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Now we can import from crewai_approach without modifying its files
from crewai_approach.tools.repo_analyzer_tool import RepoAnalyzerTool
from crewai_approach.tools.pattern_analyzer_tool import PatternAnalyzerTool
from crewai_approach.tools.directory_analyzer_tool import DirectoryAnalyzer
from crewai_approach.tools.repo_metrics_tool import RepositoryMetricsCalculator

logger = logging.getLogger(__name__)

class AnalysisService:
    """Service for handling repository analysis operations"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the analysis service"""
        self.config = config or {}
        # Tools are now initialized within methods that receive repo_path

    async def analyze_repository(self,
                               repo_path: str,
                               max_files: Optional[int] = None,
                               max_batch_size: Optional[int] = None,
                               verbose: int = 0,
                               manager_llm: Optional[str] = None): # Changed return type to yield TextContent
        """
        Analyze a repository by running the crewai_approach/run_crew_pr.py script.

        Args:
            repo_path: Path to the repository
            max_files: Maximum number of changed files to analyze fully.
            max_batch_size: Target maximum files per processing batch.
            verbose: Increase verbosity level (0, 1, or 2).
            manager_llm: LLM model to use for the manager agent/process.

        Yields:
            TextContent objects containing the script's standard output and error.
        Returns:
            Dictionary containing analysis results from the script's JSON output file.
        """
        temp_dir = None
        try:
            logger.info(f"Running crewai_approach/run_crew_pr.py for: {repo_path}")
            yield TextContent(type="text", text=f"Running analysis for {repo_path}...\n")

            # Create a temporary directory for output
            temp_dir = tempfile.mkdtemp()
            temp_output_path = Path(temp_dir)

            # Construct the command to run the script
            # The script saves output to a file specified by --output-dir
            command = [
                sys.executable, # Use the current Python executable
                "crewai_approach/run_crew_pr.py",
                repo_path,
                "--output-dir", str(temp_output_path),
            ]

            # Add optional arguments if provided
            if max_files is not None:
                command.extend(["--max-files", str(max_files)])
            if max_batch_size is not None:
                command.extend(["--max-batch-size", str(max_batch_size)])
            if verbose > 0:
                # Add -v flags based on the verbose level
                command.extend(["--verbose"])
            if manager_llm is not None:
                command.extend(["--manager-llm", manager_llm])

            # Execute the command as a subprocess
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                # Set PYTHONPATH and OPENAI_API_KEY for the subprocess
                env={
                    "PYTHONPATH": ".",
                    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "") # Pass the API key from the current environment
                }
            )

            # Read stdout and stderr concurrently and stream
            async def stream_output(stream, stream_name):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    yield TextContent(type="text", text=line.decode().strip() + "\n")

            stdout_task = asyncio.create_task(self._collect_stream(stream_output(process.stdout, "stdout")))
            stderr_task = asyncio.create_task(self._collect_stream(stream_output(process.stderr, "stderr")))

            # Yield output as it becomes available
            for task in asyncio.as_completed([stdout_task, stderr_task]):
                 for content in await task:
                     yield content

            # Wait for the process to complete
            await process.wait()

            if process.returncode != 0:
                error_message = stderr_task.result() if stderr_task.result() else "Unknown script error"
                logger.error(f"Script execution failed with return code {process.returncode}: {error_message}")
                yield TextContent(type="text", text=f"Error: Script execution failed with return code {process.returncode}: {error_message}\n")
                # Propagate the error instead of returning a value
                raise Exception(f"Script execution failed: {error_message}")

            # The script saves output to a file named *_final_recommendations.json
            # Find the generated JSON file in the temporary directory
            output_files = list(temp_output_path.glob("*_final_recommendations.json"))

            if not output_files:
                logger.error(f"No output JSON file found in {temp_dir}")
                yield TextContent(type="text", text="Error: Analysis script did not produce an output file.\n")
                # Propagate the error instead of returning a value
                raise FileNotFoundError("Analysis script did not produce an output file.")

            # Assuming there's only one such file
            output_file_path = output_files[0]

            # Read and parse the JSON output file
            try:
                with open(output_file_path, 'r') as f:
                    analysis_result = json.load(f)
                logger.info("Script executed successfully and output JSON parsed.")
                yield TextContent(type="text", text="Analysis completed successfully.\n")
                # Yield the path to the output file as a distinct indicator
                yield TextContent(type="text", text=f"OUTPUT_FILE_PATH: {output_file_path}\n")
                # The final analysis result will be returned implicitly by the async generator
                # when it finishes yielding.
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON output file '{output_file_path}': {e}")
                yield TextContent(type="text", text=f"Error: Failed to parse script output file: {str(e)}\n")
                # Propagate the error instead of returning a value
                raise json.JSONDecodeError(f"Failed to parse script output file: {str(e)}", e.doc, e.pos) from e

        except FileNotFoundError:
            logger.error(f"Script not found: crewai_approach/run_crew_pr.py")
            yield TextContent(type="text", text="Error: Analysis script not found.\n")
            # Propagate the error instead of returning a value
            raise FileNotFoundError("Analysis script not found.")
        except Exception as e:
            logger.error(f"Error during script execution or file handling: {e}", exc_info=True)
            yield TextContent(type="text", text=f"Error: Repository analysis failed: {str(e)}\n")
            # Propagate the error instead of returning a value
            raise Exception(f"Repository analysis failed: {str(e)}") from e
        finally:
            # Clean up the temporary directory
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
                yield TextContent(type="text", text=f"Cleaned up temporary directory: {temp_dir}\n")

    async def _collect_stream(self, stream_generator):
        """Collects items from an async generator into a list."""
        # This method is used internally by analyze_repository to collect stream output.
        # It should not be called directly by the client.
        return [item async for item in stream_generator]


    async def analyze_files(self, file_paths: List[str], repo_path: str) -> Dict[str, Any]:
        """
        Analyze specific files

        Args:
            file_paths: List of file paths to analyze
            repo_path: Path to the repository

        Returns:
            Dictionary containing file analysis results
        """
        results = {}
        # Initialize PatternAnalyzerTool within the method
        pattern_analyzer = PatternAnalyzerTool(repo_path=repo_path)

        # PatternAnalyzerTool._run expects file_paths and repository_path
        # It does not have an analyze_file method based on the previous file content.
        # Let's call its _run method with the list of files.
        try:
            pattern_results_json = pattern_analyzer._run(file_paths=file_paths, repository_path=repo_path)
            results['file_patterns'] = json.loads(pattern_results_json)
        except Exception as e:
            logger.error(f"Error analyzing files {file_paths}: {e}", exc_info=True)
            results['file_patterns'] = {"error": f"File pattern analysis failed: {str(e)}"}

        # Note: The original analyze_files seemed to loop through files and call pattern_analyzer.analyze_file.
        # Based on the PatternAnalyzerTool content, there is no analyze_file method.
        # The _run method takes a list of file_paths.
        # The logic here is adjusted to call _run once with the list.
        # If individual file analysis is needed, the PatternAnalyzerTool would need modification.

        return results


    async def get_repository_summary(self, repo_path: str) -> Dict[str, Any]:
        """
        Get a summary of the repository

        Args:
            repo_path: Path to the repository

        Returns:
            Dictionary containing repository summary
        """
        try:
            # Initialize tools within the method
            repo_analyzer = RepoAnalyzerTool(repo_path=repo_path)
            metrics_calculator = RepositoryMetricsCalculator(repo_path=repo_path)

            # Quick summary using repo analyzer
            # RepoAnalyzerTool._run returns RepositoryAnalysis JSON string
            repo_analysis_json = repo_analyzer._run()
            repo_analysis_data = json.loads(repo_analysis_json)

            # Add basic metrics
            # RepositoryMetricsCalculator._run expects repository_analysis_json
            metrics_json = metrics_calculator._run(repository_analysis_json=repo_analysis_json)
            metrics_data = json.loads(metrics_json)

            # Combine relevant parts for a summary
            summary = {
                "repo_path": repo_analysis_data.get("repo_path", repo_path),
                "total_files_changed": repo_analysis_data.get("total_files_changed", 0),
                "total_lines_changed": repo_analysis_data.get("total_lines_changed", 0),
                "file_extensions": repo_analysis_data.get("extensions_summary", {}),
                "basic_metrics": {
                    "avg_lines_per_file": metrics_data.get("change_metrics", {}).get("avg_lines_per_file", 0.0),
                    "directory_count": metrics_data.get("directory_metrics", {}).get("directory_count", 0),
                    "file_type_count": metrics_data.get("file_type_metrics", {}).get("file_type_count", 0)
                },
                "analysis_summary": repo_analysis_data.get("analysis_summary", "Basic analysis completed.")
            }

            return summary

        except Exception as e:
            logger.error(f"Error getting repository summary: {e}", exc_info=True)
            # Return a structured error response
            return {"error": f"Repository summary failed: {str(e)}"}


    async def analyze_files(self, file_paths: List[str], repo_path: str) -> Dict[str, Any]:
        """
        Analyze specific files

        Args:
            file_paths: List of file paths to analyze
            repo_path: Path to the repository

        Returns:
            Dictionary containing file analysis results
        """
        results = {}
        # Initialize PatternAnalyzerTool within the method
        pattern_analyzer = PatternAnalyzerTool(repo_path=repo_path)

        # PatternAnalyzerTool._run expects file_paths and repository_path
        # It does not have an analyze_file method based on the previous file content.
        # Let's call its _run method with the list of files.
        try:
            pattern_results_json = pattern_analyzer._run(file_paths=file_paths, repository_path=repo_path)
            results['file_patterns'] = json.loads(pattern_results_json)
        except Exception as e:
            logger.error(f"Error analyzing files {file_paths}: {e}", exc_info=True)
            results['file_patterns'] = {"error": f"File pattern analysis failed: {str(e)}"}

        # Note: The original analyze_files seemed to loop through files and call pattern_analyzer.analyze_file.
        # Based on the PatternAnalyzerTool content, there is no analyze_file method.
        # The _run method takes a list of file_paths.
        # The logic here is adjusted to call _run once with the list.
        # If individual file analysis is needed, the PatternAnalyzerTool would need modification.

        return results


    async def get_repository_summary(self, repo_path: str) -> Dict[str, Any]:
        """
        Get a summary of the repository

        Args:
            repo_path: Path to the repository

        Returns:
            Dictionary containing repository summary
        """
        try:
            # Initialize tools within the method
            repo_analyzer = RepoAnalyzerTool(repo_path=repo_path)
            metrics_calculator = RepositoryMetricsCalculator(repo_path=repo_path)

            # Quick summary using repo analyzer
            # RepoAnalyzerTool._run returns RepositoryAnalysis JSON string
            repo_analysis_json = repo_analyzer._run()
            repo_analysis_data = json.loads(repo_analysis_json)

            # Add basic metrics
            # RepositoryMetricsCalculator._run expects repository_analysis_json
            metrics_json = metrics_calculator._run(repository_analysis_json=repo_analysis_json)
            metrics_data = json.loads(metrics_json)

            # Combine relevant parts for a summary
            summary = {
                "repo_path": repo_analysis_data.get("repo_path", repo_path),
                "total_files_changed": repo_analysis_data.get("total_files_changed", 0),
                "total_lines_changed": repo_analysis_data.get("total_lines_changed", 0),
                "file_extensions": repo_analysis_data.get("extensions_summary", {}),
                "basic_metrics": {
                    "avg_lines_per_file": metrics_data.get("change_metrics", {}).get("avg_lines_per_file", 0.0),
                    "directory_count": metrics_data.get("directory_metrics", {}).get("directory_count", 0),
                    "file_type_count": metrics_data.get("file_type_metrics", {}).get("file_type_count", 0)
                },
                "analysis_summary": repo_analysis_data.get("analysis_summary", "Basic analysis completed.")
            }

            return summary

        except Exception as e:
            logger.error(f"Error getting repository summary: {e}", exc_info=True)
            # Return a structured error response
            return {"error": f"Repository summary failed: {str(e)}"}
