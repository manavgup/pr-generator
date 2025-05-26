**Summary**

This project provides a modular FastAPI service that generates structured pull-request descriptions by orchestrating an LLM-driven pipeline via JSON-RPC and Server-Sent Events, fully conforming to the Agent-to-Agent (A2A) protocol with built-in authentication, discovery, and progress tracking  ￼.

⸻

**Introduction**
```bash
	•	The service exposes a FastAPI HTTP API for pull-request description generation, leveraging Python type hints and ASGI performance characteristics  ￼.
	•	It implements the JSON-RPC 2.0 protocol to receive structured method calls and return results or errors in a standardized format  ￼.
	•	Real-time progress is published over Server-Sent Events (SSE), enabling clients to receive incremental task updates without polling  ￼.
	•	The agent advertises its capabilities via a WebFinger discovery endpoint and an Agent Card, per the A2A specification  ￼ ￼.
```
⸻

**Features**
```bash
	•	Pull-Request Summary Generation: Invokes an IBM Watsonx LLM to produce a <|python_tag|> tool call, which is executed to generate a detailed PR description.
	•	Progress Bar: Uses tqdm to display a ten-step progress bar based on “Defining task:” logs from the backend pipeline  ￼.
	•	Strong Typing: Validates all RPC requests and responses with Pydantic v2 models for schemas and data integrity  ￼.
	•	Agent-to-Agent Protocol Compliance:
	•	Agent Card at /.well-known/agent.json
	•	WebFinger at /.well-known/webfinger
	•	JSON-RPC at /rpc
	•	SSE Streaming at /rpc/stream  ￼.
	•	Authentication: Secures endpoints with an X-API-Key header, validated against environment variables.
```
⸻

**Installation**
```bash
	1.	Clone the repository
```
git clone https://github.com/manavgup/pr-generator-2.git
cd pr-generator-2

```bash
	2.	Create and activate a virtual environment
```

python3 -m venv .venv
source .venv/bin/activate

```bash
	3.	Install dependencies
```

pip install -e .
pip install -r requirements.txt



⸻

**Configuration**

Place your credentials in a .env file at the project root, for example:
```bash
WATSONX_URL=<your_ibm_watsonx_url>
WATSONX_API_KEY=<your_ibm_api_key>
WATSONX_PROJECT_ID=<your_project_id>
CHROMA_OPENAI_API_KEY=<your_openai_key>
```

The service will auto-load these variables at startup via python-dotenv.

⸻

**Usage**

	1.	Start the server
```bash
uvicorn a2a.server:app --host 0.0.0.0 --port 8200 --reload
```

	2.	Invoke JSON-RPC
```bash
curl -X POST http://127.0.0.1:8200/rpc \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $CHROMA_OPENAI_API_KEY" \
  --data '{"jsonrpc":"2.0","id":1,"method":"recommendPR","params":{"repo_path":"/path/to/repo"}}'
```

	3.	Stream progress via SSE
```bash
Connect to http://127.0.0.1:8200/rpc/stream with an EventSource, ensuring you include the same X-API-Key header.
```
⸻

**Architecture**
```bash
	•	LLM Integration: Uses langchain_ibm.ChatWatsonx for requests and responses, with greedy decoding and configurable token limits.
	•	CrewAI CLI: Delegates the core PR-generation logic to crewai_approach.run_crew_pr as a subprocess, capturing its ten-step internal workflow.
	•	Regex Dispatch: Detects tool-invocation strings (git_diff, requests.get, or custom calls) in the LLM output and routes them to the subprocess handler.
	•	Temporary Cloning: For remote repos, performs a shallow git clone into a temp directory; for local paths, operates in-place.
```
⸻

**Endpoints**
```bash
Path	Method	Description
/.well-known/agent.json	GET	Agent Card with metadata and capabilities
/.well-known/webfinger	GET	WebFinger discovery following A2A spec
/rpc	POST	JSON-RPC 2.0 method call for recommendPR
/rpc/stream	GET	SSE stream of progress events (start, progress, complete)
```

⸻

License
MIT License.

⸻
