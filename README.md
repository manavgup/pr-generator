# PR Generator

A modular AI-powered system for intelligently analyzing Git changes and generating pull request (PR) recommendations using different strategies — including LLM-guided reasoning, CrewAI agents, and TF-IDF similarity.

---

## 🚀 Features

- Automatically detects unstaged Git changes in a repository
- Groups related files into logical PRs with:
  - 📝 Titles
  - 📄 Descriptions
  - 💡 Reasoning
- Supports multiple PR generation strategies:
  - **CrewAI Agentic Analysis**
  - **LLM-Guided Grouping**
  - **TF-IDF Similarity Clustering**
- Integrates with multiple LLM providers (OpenAI, Ollama)
- Uses standardized, strongly typed Pydantic 2.0 models
- Outputs structured JSON results for each PR suggestion
- Modular architecture for easy extension and experimentation

---

## 🧠 Approaches

### 1. `crewai_approach/` (Primary Focus)
Uses CrewAI to orchestrate a team of agents that analyze and group code changes into PRs. Agents perform tasks like code analysis, pattern recognition, validation, and content generation.

### 2. `llm_guided_approach/` *(Legacy/Experimental)*
Uses custom LLM pipelines and generators to group files and generate PR content.

### 3. `tf_idf/` *(Legacy/Experimental)*
Uses traditional NLP techniques (TF-IDF + cosine similarity) to find related file changes.

---

## 🗂️ Project Structure (Simplified)

```
pr-generator/
├── crewai_approach/            # CrewAI-powered PR generation
│   ├── tools/                  # Modular tools for file analysis and grouping
│   ├── models/                 # Pydantic 2.0 data models
│   ├── config/                 # Agent/task configuration in YAML
│   ├── test_*.py               # Test files for toolchain and logic
│   ├── crew.py                 # Crew definition and setup
│   └── run_crew_pr.py          # Script entry point
│
├── llm_guided_approach/        # Legacy approach using LLM-only logic
│   ├── generators/             # Grouping and PR generation logic
│   ├── llm_service.py
│   └── scripts/generate_prs.py
│
├── tf_idf/                     # Traditional NLP-based grouping
│   └── tf_idf.py
│
├── shared/                     # Shared models and utilities
│   ├── config/
│   ├── models/
│   ├── tools/
│   └── utils/
│
├── embeddings/                 # Embedding-based similarity analysis
├── output/                     # Output from runs (structured groupings)
├── docs/                       # Design notes and architectural documentation
├── generate_graph.py           # PR graph visualizer
├── compare.py                  # Compare outputs from different strategies
└── README.md                   # This file
```

---

## 🧩 Setup

### Python Version

- Python 3.8+

### Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### Install Dependencies (for `crewai_approach/`)

```bash
pip install -r crewai_approach/requirements.txt
```

### Environment Variables

For OpenAI usage:

```bash
export OPENAI_API_KEY=your_openai_key
```

For Ollama usage:

```bash
# Default URL is http://localhost:11434
export OLLAMA_API_URL=http://localhost:11434
```

---

## 🧪 Example Usage (CrewAI)

```bash
# Run CrewAI-based PR generator on a Git repo
python crewai_approach/run_crew_pr.py /path/to/your/repo --max-files 30 --provider openai
```

Optional flags:

- `--provider openai|ollama` – Choose LLM backend
- `--max-files` – Max files per PR group
- `--model` – LLM model to use (e.g., `gpt-4`, `llama3`)
- `--output` – Path to save JSON output
- `--verbose` – Enable detailed logging

---

## 🧠 Agent Architecture (CrewAI)

- **Repository Analyzer Tool**: Scans for changed files, computes metrics.
- **Pattern Analyzer Tool**: Extracts filename/pattern similarities.
- **Batch Splitter Tool**: Splits files into manageable chunks for analysis.
- **Grouping Strategy Selector Tool**: Determines which grouping strategy to use.
- **Group Refiner/Validator Tools**: Improves and validates group quality.
- **PR Crew**: Orchestrates agents and passes structured data between tools.

---

## 📦 Output Format

Each generated PR suggestion includes:

```json
{
  "title": "feat(core): improve auth",
  "description": "Enhanced auth flow...",
  "reasoning": "These files all deal with authentication middleware.",
  "files": [
    "backend/core/auth_middleware.py",
    "backend/core/token_utils.py"
  ]
}
```

---

## ✅ Tests

To run tests (example: for CrewAI tools):

```bash
pytest crewai_approach/
```

---

## 📌 Roadmap

- [ ] Add LangGraph-style composable agents (under `langgraph_approach/`)
- [ ] Web-based visualization for PR suggestions
- [ ] Fine-tuned LLM models for code grouping
- [ ] Human-in-the-loop feedback loop

---

## 🧑‍💻 Contributing

If you're collaborating specifically on `crewai_approach/`, please refer to its internal `requirements.txt` and test files. Contributions are welcome via PRs.

---

## 🪪 License

MIT License