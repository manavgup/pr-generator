# TF-IDF-Based PR Grouper

This module automatically categorizes changed files in a Git repository into meaningful PR (Pull Request) groups using a combination of **TF-IDF feature extraction** and **HDBSCAN clustering**.

It analyzes:
- **Git history** to identify file co-occurrence patterns.
- **File structure and naming** using TF-IDF (Term Frequency-Inverse Document Frequency).
- Applies **clustering** to group files and generate **intelligent labels** for each group.

---

## 📦 Features
- Uses **TF-IDF** to extract weighted features from file paths and names.
- Applies **HDBSCAN** clustering to group related files.
- Performs **secondary clustering** for large groups (>30 files).
- Generates **readable labels** like `IntegrationTest Ingestion Py DeepModule`.

---
## High level flowchart

```
flowchart TD
    A[Start Script] --> B[Parse Arguments<br>Get Repo Path]
    B --> C[Validate Git Repo]
    C --> D[Get Historical Co-occurrence Matrix<br>From git log]
    D --> E[Get Changed Files<br>From git status]
    E --> F{Changed Files Exist?}
    F -- No --> Z[Exit - No Changes Detected]
    F -- Yes --> G[Extract Structural Features]
    G --> H[Compute Co-occurrence Features]
    H --> I[Combine Features]
    I --> J[Cluster Files with HDBSCAN]

    J --> K[Group Files by Cluster ID]
    K --> L{Large Cluster?}
    L -- No --> M[Add to Final Clusters]
    L -- Yes --> N[Secondary Cluster<br>HDBSCAN with finer params]
    N --> O[Group Files by Sub-Cluster]
    O --> P[Add Sub-Clusters to Final Clusters]
    M & P --> Q[Generate Labels for Final Clusters]
    Q --> R[Print Results with Labels & File Lists]
    R --> S[End Script]
```

## 🚀 How to Run

```bash
python tf_idf/tf_idf.py --path /path/to/your/git/repo
```

Example:
```bash
python tf_idf/tf_idf.py --path ../my-repo > output/tf_idf/run_2025-03-18.txt
```

---

## 📊 Example Output

```
Analyzing git history in /path/to/repo...

Auto-generated PR Categories for /path/to/repo:

=== Test Ingestion Py DeepModule ===
Files (5):
  backend/tests/integration/test_ingestion.py
  backend/tests/integration/test_data_ingestion.py
  ...

=== Models Services Py MidModule ===
Files (3):
  backend/models/user.py
  backend/services/user_service.py
  ...
```

You can find a full example output in [`examples/sample_output.txt`](examples/sample_output.txt).

---

## 📁 File Overview

| File                | Purpose                                           |
|---------------------|---------------------------------------------------|
| `tf_idf.py`         | Main script to perform PR grouping with TF-IDF    |
| `examples/`         | Example output files for reference                 |

---

## 🛠️ Dependencies
Make sure you have the following installed in your Python environment:
- `scikit-learn`
- `hdbscan`
- `nltk`
- `numpy`

Install via pip:
```bash
pip install -r requirements.txt
```

_Note: The script downloads NLTK stopwords on first run._

---

## 🧭 What's Next?
This is the **first iteration** in a series of approaches:
- ✅ TF-IDF-based grouping (this version)
- 🚧 BERT embeddings-based grouping (coming soon)
- 🚧 Agentic AI approach with CrewAI/LangGraph (planned)

---

## 📜 License
MIT License
