#!/usr/bin/env python
import sys
from pathlib import Path
import subprocess
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv()
from shared.utils.logging_utils import configure_logging, get_logger
configure_logging(verbose=False)
logger = get_logger(__name__)
import os
chroma_key = os.getenv("CHROMA_OPENAI_API_KEY")
if not chroma_key:
    logger.error("CHROMA_OPENAI_API_KEY is not set")
    sys.exit(1)
# ensure OPENAI API key is set for LLM calls
openai_key = os.getenv("OPENAI_API_KEY") or chroma_key
if not openai_key:
    logger.error("Both OPENAI_API_KEY and CHROMA_OPENAI_API_KEY are not set")
    sys.exit(1)
# propagate to subprocesses and libraries
os.environ["OPENAI_API_KEY"] = openai_key
# ensure downstream code and CrewAI sees the key
os.environ["CHROMA_OPENAI_API_KEY"] = chroma_key
"""
Hierarchical runner for the PR recommendation system, with a
PR Orchestrator manager agent that:
  1. Splits large diffs into chunks (ChunkRouterAgent)
  2. Summarises each chunk (DiffSummariserAgent), refusing >8k tokens
  3. Validates each stage before moving on
  4. Delegates final grouping to your existing SequentialPRCrew
"""

# import your existing crew and any CrewAI primitives you need
from crewai_approach.crew import SequentialPRCrew
try:
    from crewai_approach.process import Process, CrewBase, Task, before_task
except ImportError:
    # Fallback if process.py isn't installed as a package
    from run_crew_pr import Process, CrewBase, Task, before_task

TOKEN_LIMIT = 8000  # adjust as needed


class ChunkRouterAgent(CrewBase):
    """Slice the full git diff into at most `max_chunk_lines` lines each."""
    @Task(name="chunk_diff")
    def chunk_diff(
        self,
        repo_path: str,
        max_chunk_lines: int = 500,
    ) -> list[str]:
        full_diff = subprocess.run(
            ["git", "diff", "HEAD"], cwd=repo_path, stdout=subprocess.PIPE, text=True, check=True
        ).stdout
        lines = full_diff.splitlines(keepends=True)
        chunks = []
        for i in range(0, len(lines), max_chunk_lines):
            chunks.append("".join(lines[i:i + max_chunk_lines]))
        return chunks


class DiffSummariserAgent(CrewBase):
    """Summarise a single diff chunk into an 8-line markdown snippet."""
    @Task(name="summarise_diff")
    def summarise(
        self,
        diff_chunk: str,
    ) -> str:
        # here you’d call your LLM; placeholder below:
        summary = "\n".join(diff_chunk.splitlines()[:8])
        if len(summary) > TOKEN_LIMIT:
            raise ValueError(f"Chunk summary exceeds {TOKEN_LIMIT} tokens")
        return summary


class PROrchestratorAgent(CrewBase):
    """Manager agent that invokes router, summariser, then SequentialPRCrew."""
    allow_delegation = True

    @before_task("orchestrate")
    def init_state(self, inputs: dict):
        self.router = ChunkRouterAgent()
        self.summariser = DiffSummariserAgent()
        self.crew = SequentialPRCrew(
            repo_path=inputs["repo_path"],
            max_files=inputs.get("max_files", 50),
            max_batch_size=inputs.get("max_batch_size", 10),
            verbose=inputs.get("verbose", 0),
            output_dir=inputs.get("output_dir", "outputs"),
            manager_llm_name=inputs.get("manager_llm_name", "gpt-4o")
        )

    @Task(name="orchestrate")
    def orchestrate(self, repo_path: str) -> dict:
        logger.info("[Orchestrator] chunking diff")
        chunks = self.router.chunk_diff(repo_path)
        logger.info(
            f"[Orchestrator] produced {len(chunks)} chunks"
        )

        summaries = []
        for idx, chunk in enumerate(chunks):
            logger.info(f"[Orchestrator] summarising chunk {idx+1}/{len(chunks)}")
            summ = self.summariser.summarise(chunk)
            summaries.append(summ)

        logger.info("[Orchestrator] delegating to SequentialPRCrew")
        # pass the summaries instead of full diff
        crew_inputs = {
            "repo_path": repo_path,
            "diff_summaries": summaries,
            "max_files": self.crew.max_files,
            "max_batch_size": self.crew.max_batch_size,
            "verbose": self.crew.verbose,
            "output_dir": str(self.crew.output_dir),
            "manager_llm_name": self.crew.manager_llm_name,
        }
        result = self.crew.crew().kickoff(inputs=crew_inputs)
        return result


def main():
    parser = Process.Parser(
        description="Hierarchical PR recommendation with orchestrator"
    )
    parser.add_argument("repo_path", help="Path to your git repo")
    parser.add_argument("--max-files", type=int, default=50)
    parser.add_argument("--max-batch-size", type=int, default=10)
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--manager-llm", default="gpt-4o")

    args = parser.parse_args()
    configure_logging(verbose=(args.verbose > 0))
    inputs = vars(args)

    # validate repo
    repo = Path(inputs["repo_path"])
    if not (repo / ".git").is_dir():
        logger.error("Not a git repo: %s", repo)
        sys.exit(1)

    orchestrator = PROrchestratorAgent()
    orchestrator.init_state(inputs)
    logger.info("Starting hierarchical orchestrated run")
    # Directly invoke the orchestrator without Process.hierarchical
    result = orchestrator.orchestrate(inputs["repo_path"])
    # Ensure output directory exists
    output_dir = Path(inputs["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save the hierarchical run result as JSON
    import json
    output_file = output_dir / "hierarchical_results.json"
    with open(output_file, "w") as f:
        # Convert Pydantic CrewOutput to a serializable dict before dumping
        if hasattr(result, "dict"):
            data = result.dict()
        else:
            data = result
        json.dump(data, f, indent=2)
    logger.info("✅ Completed; results saved to %s", output_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
