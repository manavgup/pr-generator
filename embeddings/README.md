
# Embeddings-Based PR Grouping Approach

This module uses **BERT-style embeddings** and **similarity analysis** to cluster modified files into **suggested PR groups**. It builds upon the TF-IDF approach and compares three strategies for analyzing code changes, aiming to optimize how files are grouped for better pull request (PR) organization.

---

## 📊 Approach Overview

We compare **three strategies** for analyzing repository changes using BERT embeddings and clustering:

1. **All Files**  
   - Analyzes all tracked files in the repository.
2. **Changed Files (Full Content)**  
   - Uses full content of only modified files (uncommitted).
3. **Changed Files (Diff Only)**  
   - Uses only the diffs (actual changes) of modified files.

---

## 🚀 Execution

Run the analysis:

```bash
python embeddings/repo_similarity_analyzer.py /path/to/your/repo --compare-all --output-dir output
```

This will generate:
- **Hierarchical Clustering Dendrograms**
- **PR Groups Visualizations**
- **Similarity Networks**
- **t-SNE Visualizations**
- **Comparison Report** (`comparison_report.md`)

---

## Example Output

###  Analysis Statistics

| Analysis Type         | Files Analyzed | Avg Similarity | Most Similar Files                    |
|-----------------------|----------------|----------------|--------------------------------------|
| All Files             | 235            | 0.2600         | `carbon-overrides.scss` & `global.scss` (0.9965) |
| Changed Files (Full)  | 71             | **0.4716**     | `team_service.py` & `user_team_service.py` (0.9394) |
| Changed Files (Diff)  | 71             | 0.3750         | `anthropic.py` & `openai.py` (1.0000) |

- **Full content** of changed files yielded the highest average similarity, suggesting changes are focused within a functional domain.
- **Diff-only analysis** helps identify tightly coupled changes, while full content shows broader contextual relationships.

---

### 📦 Output Files

| Directory                               | Contents                                                   |
|----------------------------------------|------------------------------------------------------------|
| `output/similarity_analysis/all_files` | Analysis of entire codebase structure                      |
| `output/similarity_analysis/changed_files_full` | Full content similarity + PR groups                        |
| `output/similarity_analysis/changed_files_diff_only` | Diff-only similarity + PR groups                        |
| `output/similarity_analysis/comparison` | PR size comparisons, Venn diagrams, summary report         |

---
#### PR Groupings Comparison (Threshold 0.7)

**changed_files_full**:
- Number of PRs: 14
- Total files in PRs: 71
- Smallest PR: 1 files
- Largest PR: 57 files
- See full groupings in: `similarity_analysis/changed_files_full/pr_groups_threshold_0.7.txt`

**changed_files_diff_only**:
- Number of PRs: 21
- Total files in PRs: 71
- Smallest PR: 1 files
- Largest PR: 47 files
- See full groupings in: `similarity_analysis/changed_files_diff_only/pr_groups_threshold_0.7.txt`


### Example Visualizations

[`PR Group Comparison`](examples/pr_group_sizes_comparison_0.7.png)
[`TS NE Visualization`](examples/tsne_visualization.png)

## 📌 Conclusion

Embedding-based analysis offers deep insights into code similarity.
There is no clear decision on which grouping to use. 
Division of files between PRs is not clear.
---

## 📂 Repository Structure

```bash
embeddings/
  └── repo_similarity_analyzer.py  # Main script
output/
  └── similarity_analysis/         # Generated results
      ├── all_files/
      ├── changed_files_full/
      ├── changed_files_diff_only/
      └── comparison/
```

---
