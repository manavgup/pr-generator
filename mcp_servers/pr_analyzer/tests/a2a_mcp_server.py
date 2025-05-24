import os
import re
import sys
import tempfile
import shutil
import glob
import subprocess
import json
from typing import Any, Dict

from fastmcp import FastMCP, Client, Context
from tqdm import tqdm
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables
load_dotenv()
chroma_key = os.getenv("CHROMA_OPENAI_API_KEY")
if chroma_key:
    os.environ["OPENAI_API_KEY"] = chroma_key
WATSONX_URL = os.getenv("WATSONX_URL")
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")

# A helper Pydantic model to define tool input
class RecommendPRParams(BaseModel):
    repo_path: str

def run_diff_to_pr(path: str) -> str:
    """
    Invoke the CrewAI diff-to-PR CLI on a local path or git URL.
    Returns the final recommendation text.
    """
    # If it's a URL, clone first
    if path.startswith("http"):
        tmp = tempfile.mkdtemp()
        subprocess.run(["git", "clone", "--depth", "1", path, tmp], check=True)
        target = tmp
    else:
        target = path

    try:
        cmd = [sys.executable, "-m", "crewai_approach.run_crew_pr", target]
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
            pbar = tqdm(total=10, desc="PR Generation Tasks", unit="task")
            recommendation = ""
            for line in proc.stdout:
                print(line, end="")
                if "Defining task:" in line:
                    pbar.update(1)
                recommendation = ""  # streamed above
            proc.wait()
            # capture last JSON output if any
            outputs = glob.glob(os.path.join(os.getcwd(), "outputs", "*.json"))
            if outputs:
                latest = max(outputs, key=os.path.getmtime)
                recommendation = open(latest).read()
            pbar.close()
        return recommendation
    finally:
        if path.startswith("http"):
            shutil.rmtree(tmp)

def a2a_recommend_pr(params: RecommendPRParams, context: Context) -> Dict[str, Any]:
    """
    MCP tool implementation for recommendPR.
    """
    repo_path = params.repo_path
    print(f"[A2A] recommend_pr called with repo_path={repo_path}")

    # Invoke the LLM first (simulate ChatWatsonx.invoke)
    # Here you would integrate langchain_ibm.ChatWatsonx if desired;
    # for simplicity assume it returns a python_tag invocation
    python_tag = f"<|python_tag|>git_diff.call(query=\"{repo_path}\")"
    print(f"[A2A] LLM returned: {python_tag}")

    if python_tag.startswith("<|python_tag|>"):
        # extract path from invocation
        code = python_tag[len("<|python_tag|>"):]
        m = re.search(r'git_diff\.call\(query="(?P<path>[^"]+)"\)', code)
        if m:
            raw = m.group("path")
            # run the actual diff-to-PR process
            recommendation = run_diff_to_pr(raw)
            return {"pr_text": recommendation}
    # fallback: echo
    return {"pr_text": python_tag}

# Instantiate the MCP server and register the tool
a2a_server = FastMCP(name="A2A PR Recommender", instructions="Generate PR descriptions via diff-to-PR CLI")
a2a_server.add_tool(
    fn=a2a_recommend_pr,
    name="recommend_pr",
    description="Generate a pull-request description from a git diff"
)

if __name__ == "__main__":
    import asyncio

    # For local debugging: allow passing a repo path or use a default
    async def demo():
        # Determine repository path from command line or fallback to cwd
        repo_arg = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
        print(f"Using repository path: {repo_arg}")
        async with Client(a2a_server) as client:
            # Pass parameters nested under 'params' to match RecommendPRParams schema
            res = await client.call_tool("recommend_pr", {"params": {"repo_path": repo_arg}})
            print("Recommendation:", res)

    asyncio.run(demo())