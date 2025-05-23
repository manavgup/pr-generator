import json
import os
import re
import sys
import tempfile
import subprocess
import shutil
import glob
from fastapi import FastAPI, Request, Depends, HTTPException, Security
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from typing import Any, Dict
from subprocess import Popen, PIPE
from tqdm import tqdm
from sse_starlette.sse import EventSourceResponse

# Use the main() entrypoint from run_crew_pr.py for diff-to-PR logic
from crewai_approach.run_crew_pr import main as diff_to_pr
from crewai_approach.models.batching_models import BatchSplitterOutput
# LLM setup using Watsonx
from langchain_ibm import ChatWatsonx
from dotenv import load_dotenv

# Load environment variables for Watsonx credentials
load_dotenv()
# Propagate OpenAI key for Litellm from CHROMA_OPENAI_API_KEY
chroma_key = os.getenv("CHROMA_OPENAI_API_KEY")
if chroma_key:
    os.environ["OPENAI_API_KEY"] = chroma_key
url = os.getenv("WATSONX_URL")
apikey = os.getenv("WATSONX_API_KEY")
project_id = os.getenv("WATSONX_PROJECT_ID")

model_id_llama = "meta-llama/llama-3-405b-instruct"
parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 10000,
    "min_new_tokens": 1,
    "temperature": 0,
    "top_k": 1,
    "top_p": 1.0,
    "seed": 42
}

llm_llama = ChatWatsonx(
    model_id=model_id_llama,
    url=url,
    apikey=apikey,
    project_id=project_id,
    params=parameters
)

app = FastAPI(
    title="PR Recommender Agent (A2A)",
    openapi_url=None,  # disable Swagger UI
)

# API Key Authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
def get_api_key(api_key: str = Security(api_key_header)):
    expected = os.getenv("CHROMA_OPENAI_API_KEY")
    if not api_key or api_key != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return api_key

# 1) Agent Card at /.well-known/agent.json with no authentication
@app.get("/.well-known/agent.json")
async def agent_card():
    return {
        "name": "PR Recommender Agent",
        "description": "Suggests PR descriptions from Git diffs",
        "endpoints": {"rpc": "/rpc"},
        "methods": ["recommendPR"],
        "capabilities": {
            "streaming": True,
            "authentication": ["apiKey"],
            "discovery": ["webfinger"]
        }
    }

# 1b) WebFinger Discovery Endpoint
@app.get("/.well-known/webfinger")
async def webfinger(resource: str, request: Request):
    return {
        "subject": resource,
        "links": [
            {
                "rel": "urn:ietf:params:xml:ns:lnkd:agent",
                "href": f"{request.base_url}.well-known/agent.json"
            }
        ]
    }

# 2) JSON-RPC request/response models
class RPCRequest(BaseModel):
    jsonrpc: str
    id: Any
    method: str
    params: Dict[str, Any]

class RPCResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: Any
    result: Any = None
    error: Any = None

@app.post("/rpc", dependencies=[Depends(get_api_key)])
async def rpc(request: Request):
    payload = await request.json()
    rpc_req = RPCRequest(**payload)
    print(f"[RPC] Received request: method={rpc_req.method}, params={rpc_req.params}")

    if rpc_req.method == "recommendPR":
        repo_path = rpc_req.params.get("repo_path")
        if not repo_path:
            resp = RPCResponse(
                id=rpc_req.id,
                error={"code": -32602, "message": "Missing repo_path"}
            )
        else:
            print(f"[RPC] Invoking LLM for repo_path={repo_path}")
            # Invoke LLM to suggest either direct text or a tool call
            content = llm_llama.invoke(
                f"Generate a pull request description for the git diff at: {repo_path}"
            )
            # Ensure we have a plain string for content
            if hasattr(content, "content"):
                content = content.content
            print(f"[RPC] LLM returned content: {content}")

            # Detect if LLM returned a tool-invocation tag
            tool_prefix = "<|python_tag|>"
            # print(f"[RPC] Checking for tool_prefix in content")
            if isinstance(content, str) and content.startswith(tool_prefix):
                # Extract the diff path and call your diff-to-PR function
                code = content[len(tool_prefix):]
                # Attempt to catch either a requests.get URL or any tool call
                match = re.search(
                    r'requests\.get\("(?P<url>[^"]+)"\)|'
                    r'git_diff_to_pull_request_description\.call\(query="(?P<path>[^"]+)"\)|'
                    r'git_diff\("(?P<simple_path>[^"]+)"\)',
                    code
                )
                if match:
                    raw = match.group("path") or match.group("url") or match.group("simple_path")
                    # print(f"[RPC] Detected tool call, raw={raw}")
                    # If the match is a URL, clone the repo temporarily and diff against HEAD
                    if raw.startswith("http"):
                        tmp_dir = tempfile.mkdtemp()
                        # print(f"[RPC] Cloning URL: {raw} into {tmp_dir}")
                        try:
                            subprocess.run(["git", "clone", "--depth", "1", raw, tmp_dir], check=True)
                            # print(f"[RPC] Clone complete, running diff_to_pr CLI")
                            # Run the diff-to-PR CLI and capture its stdout
                            cmd = [sys.executable, "-m", "crewai_approach.run_crew_pr", tmp_dir]
                            # print(f"[RPC] Running PR-generator CLI on {tmp_dir}")
                            try:
                                # Run the PR-generator CLI and display progress based on log messages
                                with Popen(cmd, stdout=PIPE, stderr=subprocess.STDOUT, text=True) as proc:
                                    pbar = tqdm(total=10, desc="PR Generation Tasks", bar_format="{desc}: {n}/{total} [{bar}]")
                                    output_lines = []
                                    for line in proc.stdout:
                                        output_lines.append(line)
                                        print(line, end="")  # echo CLI output
                                        if "Defining task:" in line:
                                            pbar.update(1)
                                            pbar.refresh()
                                            # Log current progress between progress bar updates
                                            print(f"Task progress: {pbar.n}/{pbar.total} tasks completed")
                                    proc.wait()
                                    pbar.close()
                                    # Print the most recently generated JSON file
                                    output_files = glob.glob(os.path.join(os.getcwd(), "outputs", "*.json"))
                                    if output_files:
                                        latest_json = max(output_files, key=os.path.getmtime)
                                        print(f"Generated JSON file: {os.path.basename(latest_json)}")
                                    recommendation = ""  # final PR text is streamed above
                            except subprocess.CalledProcessError as e:
                                # print(f"[RPC] PR-generator CLI error, exit code={e.returncode}, stderr={e.stderr.strip()}")
                                recommendation = f"PR generation failed: {e.stderr.strip()}"
                        finally:
                            shutil.rmtree(tmp_dir)
                    else:
                        diff_arg = raw[len("git diff "):] if raw.startswith("git diff ") else raw
                        # Run the diff-to-PR CLI and capture its stdout
                        cmd = [sys.executable, "-m", "crewai_approach.run_crew_pr", diff_arg]
                        # print(f"[RPC] Running PR-generator CLI on {diff_arg}")
                        try:
                            # Run the PR-generator CLI and display progress based on log messages
                            with Popen(cmd, stdout=PIPE, stderr=subprocess.STDOUT, text=True) as proc:
                                pbar = tqdm(total=10, desc="PR Generation Tasks", bar_format="{desc}: {n}/{total} [{bar}]")
                                output_lines = []
                                for line in proc.stdout:
                                    output_lines.append(line)
                                    print(line, end="")  # echo CLI output
                                    if "Defining task:" in line:
                                        pbar.update(1)
                                        pbar.refresh()
                                        # Log current progress between progress bar updates
                                        print(f"Task progress: {pbar.n}/{pbar.total} tasks completed")
                                proc.wait()
                                pbar.close()
                                # Print the most recently generated JSON file
                                output_files = glob.glob(os.path.join(os.getcwd(), "outputs", "*.json"))
                                if output_files:
                                    latest_json = max(output_files, key=os.path.getmtime)
                                    print(f"Generated JSON file: {os.path.basename(latest_json)}")
                                recommendation = ""  # final PR text is streamed above
                        except subprocess.CalledProcessError as e:
                            # print(f"[RPC] PR-generator CLI error, exit code={e.returncode}, stderr={e.stderr.strip()}")
                            recommendation = f"PR generation failed: {e.stderr.strip()}"
                else:
                    recommendation = content
            else:
                # LLM returned plain text
                recommendation = content
            print(f"[RPC] Final recommendation: {recommendation}")

            resp = RPCResponse(id=rpc_req.id, result={"pr_text": recommendation})
    else:
        resp = RPCResponse(
            id=rpc_req.id,
            error={"code": -32601, "message": f"Method {rpc_req.method} not found"}
        )

    return JSONResponse(content=resp.dict())

# SSE Streaming endpoint for PR recommendation progress
@app.get("/rpc/stream", dependencies=[Depends(get_api_key)])
async def rpc_stream(request: Request):
    async def event_generator():
        # Yield start notification
        yield {"event": "start", "data": json.dumps({"message": "PR recommendation started"})}
        # Placeholder: yield progress events
        yield {"event": "progress", "data": json.dumps({"message": "analysis_started"})}
        # ... in a real implementation, yield after each CLI "Defining task" ...
        # When complete, yield the standard JSON-RPC response
        rpc_response = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"pr_text": "PR recommendation complete"},
            "error": None
        }
        yield {"event": "complete", "data": json.dumps(rpc_response)}
    return EventSourceResponse(event_generator())

# If run as main, start Uvicorn server
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8200"))
    uvicorn.run("a2a.server:app", host="0.0.0.0", port=port, reload=True)