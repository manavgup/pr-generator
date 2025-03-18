import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import subprocess
import re
from pathlib import Path
from tqdm import tqdm
import networkx as nx
import matplotlib.colors as mcolors
from collections import defaultdict, Counter

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# Set environment variable to avoid tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_file(file_path):
    """Load and return file content as string."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        # Skip binary files
        return ""

def get_git_diff(repo_path, file_path):
    """Get the diff for a single file."""
    try:
        # Get the diff for the file
        result = subprocess.run(
            ['git', '-C', repo_path, 'diff', 'HEAD', '--', file_path],
            capture_output=True, text=True, check=True
        )
        diff_output = result.stdout
        
        # Extract only the added/removed lines (starting with + or -)
        diff_lines = []
        for line in diff_output.split('\n'):
            if line.startswith('+') and not line.startswith('+++'):
                diff_lines.append(line[1:])  # Remove the + sign
            elif line.startswith('-') and not line.startswith('---'):
                diff_lines.append(line[1:])  # Remove the - sign
        
        return '\n'.join(diff_lines)
    except subprocess.CalledProcessError:
        # If the file is not tracked or has no changes
        return ""

def get_repo_files(repo_path, extensions=None, changed_only=False):
    """Get files from the repository, optionally filtered by extensions and change status.
    
    Args:
        repo_path: Path to the Git repository
        extensions: Optional comma-separated list of file extensions to include
        changed_only: If True, only return files that have been modified but not committed
    """
    try:
        if changed_only:
            # Get only files that have been modified but not committed
            result = subprocess.run(
                ['git', '-C', repo_path, 'status', '--porcelain'],
                capture_output=True, text=True, check=True
            )
            
            # Debug information
            if not result.stdout.strip():
                print("No changes detected in the repository.")
                return []
            
            # Parse the porcelain output
            files = []
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                
                try:
                    # Make sure the line is long enough
                    if len(line) < 2:
                        continue
                        
                    # Status code is the first two characters
                    status = line[:2].strip()
                    
                    # Make sure there's a file path after the status
                    if len(line) <= 3:
                        continue
                        
                    file_path = line[3:].strip()
                    
                    # Handle renamed files (contains " -> ")
                    if " -> " in file_path:
                        file_path = file_path.split(" -> ")[1]
                    
                    # Include modified (M), added (A), renamed (R), or copied (C) files
                    if status and (status[0] in ['M', 'A', 'R', 'C'] or (len(status) > 1 and status[1] in ['M'])):
                        files.append(file_path)
                except IndexError:
                    # Skip this line if there's an index error
                    print(f"Warning: Could not parse git status line: '{line}'")
                    continue
        else:
            # Get all tracked files
            result = subprocess.run(
                ['git', '-C', repo_path, 'ls-files'],
                capture_output=True, text=True, check=True
            )
            files = result.stdout.strip().split('\n')
        
        # Filter empty lines (can happen if repo is empty)
        files = [f for f in files if f]
        
        if not files:
            return []
            
        # Filter by extensions if provided
        if extensions:
            ext_list = extensions.split(',')
            files = [f for f in files if any(f.endswith(ext) for ext in ext_list)]
        
        # Convert to full paths
        full_paths = [os.path.join(repo_path, f) for f in files]
        
        # Filter out directories and non-existent files
        return [f for f in full_paths if os.path.isfile(f)]
    except subprocess.CalledProcessError as e:
        print(f"Error accessing repository: {e}")
        return []

def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into chunks with overlap."""
    if not text:
        return []
        
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
        
        # Stop if we've reached the end of the text
        if i + chunk_size >= len(words):
            break
    
    return chunks

def create_embeddings(chunks, model):
    """Create embeddings for each chunk using the specified model."""
    if not chunks:
        return np.array([])
    return model.encode(chunks)

def calculate_file_embedding(chunks_embeddings):
    """Calculate a single embedding for a file by averaging chunk embeddings."""
    if chunks_embeddings.size == 0:
        return None
    return np.mean(chunks_embeddings, axis=0)

def generate_pr_groups(valid_files, similarity_matrix, threshold=0.7):
    """Generate PR groups based on similarity matrix using a graph-based approach."""
    # Create a graph where nodes are files and edges are similarities above threshold
    G = nx.Graph()
    
    # Add nodes
    for i, file_path in enumerate(valid_files):
        file_name = os.path.basename(file_path)
        G.add_node(i, name=file_name, path=file_path)
    
    # Add edges for similarities above threshold
    for i in range(len(valid_files)):
        for j in range(i+1, len(valid_files)):
            similarity = similarity_matrix[i, j]
            if similarity >= threshold:
                G.add_edge(i, j, weight=similarity)
    
    # Find connected components (these become our PR groups)
    connected_components = list(nx.connected_components(G))
    
    # Sort components by size (descending)
    connected_components.sort(key=len, reverse=True)
    
    # Create PR groups
    pr_groups = []
    for component in connected_components:
        group = []
        for node_id in component:
            file_path = valid_files[node_id]
            group.append((file_path, node_id))
        pr_groups.append(group)
    
    # Handle isolated nodes (files with no connections above threshold)
    # Group them by file path patterns
    isolated_nodes = []
    for i in range(len(valid_files)):
        if not any(i in component for component in connected_components):
            isolated_nodes.append((valid_files[i], i))
    
    # Group isolated nodes by directory path pattern
    if isolated_nodes:
        dir_groups = defaultdict(list)
        for file_path, node_id in isolated_nodes:
            # Use the second-level directory as a grouping key
            path_parts = Path(file_path).parts
            if len(path_parts) >= 3:
                group_key = f"{path_parts[-3]}/{path_parts[-2]}"
            else:
                group_key = os.path.dirname(file_path)
            dir_groups[group_key].append((file_path, node_id))
        
        # Add directory groups to PR groups
        for dir_group in dir_groups.values():
            if dir_group:
                pr_groups.append(dir_group)
    
    return pr_groups

def generate_enhanced_pr_groups(valid_files, similarity_matrix, repo_path, threshold=0.7):
    """Generate PR groups based on similarity matrix combined with directory structure and imports."""
    # Create base graph from embeddings similarity
    G = nx.Graph()
    
    # Add nodes
    for i, file_path in enumerate(valid_files):
        file_name = os.path.basename(file_path)
        G.add_node(i, name=file_name, path=file_path)
    
    # 1. Add edges based on embedding similarity (with reduced weight)
    for i in range(len(valid_files)):
        for j in range(i+1, len(valid_files)):
            similarity = similarity_matrix[i, j]
            if similarity >= threshold:
                # Use 70% of the original similarity weight
                G.add_edge(i, j, weight=similarity * 0.7, type="semantic")
    
    # 2. Add edges based on directory structure
    for i in range(len(valid_files)):
        for j in range(i+1, len(valid_files)):
            # Get directory paths
            dir_i = os.path.dirname(valid_files[i])
            dir_j = os.path.dirname(valid_files[j])
            
            # Calculate directory similarity (how many directory levels they share)
            dir_parts_i = dir_i.split(os.sep)
            dir_parts_j = dir_j.split(os.sep)
            shared_levels = 0
            for a, b in zip(dir_parts_i, dir_parts_j):
                if a == b:
                    shared_levels += 1
                else:
                    break
            
            # Only consider files in the same or related directories
            if shared_levels >= 2:  # They share at least 2 directory levels
                # Calculate directory similarity score (0.0-1.0)
                max_levels = max(len(dir_parts_i), len(dir_parts_j))
                dir_similarity = shared_levels / max_levels
                
                # Add edge with a weight based on directory structure
                # Use 20% of the weight for directory structure
                if i in G and j in G:  # Make sure nodes exist
                    if G.has_edge(i, j):
                        # Update existing edge
                        G[i][j]['weight'] += dir_similarity * 0.2
                    else:
                        # Add new edge
                        G.add_edge(i, j, weight=dir_similarity * 0.2, type="directory")
    
    # 3. Add edges based on import dependencies
    for i, file_i in enumerate(valid_files):
        # Only process Python files
        if not file_i.endswith('.py'):
            continue
            
        # Extract imports from this file
        imports = []
        try:
            with open(file_i, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Simple regex to find import statements
                import_lines = re.findall(r'^(?:from|import)\s+([.\w]+)', content, re.MULTILINE)
                imports = [imp.split('.')[-1] for imp in import_lines]  # Get the last part of the import
        except Exception:
            pass
            
        # Look for connections based on imports
        for j, file_j in enumerate(valid_files):
            if i == j or not file_j.endswith('.py'):
                continue
                
            # Check if this file name is imported by file_i
            file_j_module = os.path.basename(file_j).replace('.py', '')
            
            if file_j_module in imports:
                # Add a strong connection for direct imports
                # Use 30% of the weight for import relationships
                if i in G and j in G:  # Make sure nodes exist
                    if G.has_edge(i, j):
                        # Update existing edge with a strong import bonus
                        G[i][j]['weight'] += 0.3
                        G[i][j]['type'] = f"{G[i][j]['type']},import"
                    else:
                        # Add new edge
                        G.add_edge(i, j, weight=0.3, type="import")
    
    # Find connected components with the enhanced graph
    connected_components = list(nx.connected_components(G))
    
    # Sort components by size (descending)
    connected_components.sort(key=len, reverse=True)
    
    # Create PR groups
    pr_groups = []
    for component in connected_components:
        group = []
        for node_id in component:
            file_path = valid_files[node_id]
            group.append((file_path, node_id))
        pr_groups.append(group)
    
    return pr_groups
def generate_hierarchical_pr_groups(valid_files, similarity_matrix, max_distance=0.4):
    """Generate PR groups using hierarchical clustering."""
    # Convert similarity matrix to distance matrix
    # (1 - similarity) gives us distance: 0 means identical, 1 means completely different
    distance_matrix = 1 - similarity_matrix
    
    # Ensure diagonal is exactly zero (fix for the error)
    np.fill_diagonal(distance_matrix, 0)
    
    # Convert square distance matrix to condensed form required by linkage
    condensed_distances = squareform(distance_matrix)
    
    # Perform hierarchical clustering using complete linkage
    # Complete linkage considers the maximum distance between all elements in two clusters
    Z = linkage(condensed_distances, method='complete')
    
    # Cut the dendrogram at the specified distance to form clusters
    # max_distance controls how similar files need to be to get clustered together
    cluster_labels = fcluster(Z, max_distance, criterion='distance')
    
    # Group files by cluster
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append((valid_files[i], i))
    
    # Convert to list of clusters and sort by size
    pr_groups = list(clusters.values())
    pr_groups.sort(key=len, reverse=True)
    
    return pr_groups

# You can also visualize the hierarchical clustering with a dendrogram
def create_dendrogram(valid_files, similarity_matrix, output_dir):
    """Create and save a dendrogram visualization of hierarchical clustering."""
    from scipy.cluster.hierarchy import dendrogram
    
    # Convert to distance matrix
    distance_matrix = 1 - similarity_matrix
    
    # Ensure diagonal is exactly zero (fix for the error)
    np.fill_diagonal(distance_matrix, 0)
    
    condensed_distances = squareform(distance_matrix)
    
    # Compute linkage
    Z = linkage(condensed_distances, method='complete')
    
    # Create large figure for readability
    plt.figure(figsize=(20, 10))
    
    # Create dendrogram
    labels = [os.path.basename(f) for f in valid_files]
    dendrogram(
        Z,
        labels=labels,
        orientation='right',
        leaf_font_size=8,
        truncate_mode='lastp',  # Show only the last p merged clusters
        p=50,  # Number of leaves to show if truncate_mode is used
        show_contracted=True,  # Show cluster sizes when truncated
    )
    
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Distance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dendrogram.png'), dpi=300, bbox_inches='tight')
    plt.close()

def analyze_repo(repo_path, model, chunk_size, overlap, extensions, changed_only, diff_only, output_dir):
    """Run a single analysis with given parameters and return results."""
    
    # Create a specific output directory for this analysis
    analysis_type = "all_files"
    if changed_only and diff_only:
        analysis_type = "changed_files_diff_only"
    elif changed_only:
        analysis_type = "changed_files_full"
    elif diff_only:
        analysis_type = "all_files_diff_only"
    
    # Add suffix to output directory
    current_output_dir = os.path.join(output_dir, analysis_type)
    os.makedirs(current_output_dir, exist_ok=True)
    
    # Get files from repository
    repo_files = get_repo_files(repo_path, extensions, changed_only)
    if not repo_files:
        print(f"No files found in the repository matching the criteria for {analysis_type}.")
        return None, []
    
    print(f"\n--- Analysis: {analysis_type} ---")
    print(f"Found {len(repo_files)} files to analyze.")
    
    # Process each file
    file_embeddings = []
    file_contents = []
    valid_files = []
    
    for file_path in tqdm(repo_files, desc="Processing files"):
        try:
            if diff_only:
                # Get only the diff
                content = get_git_diff(repo_path, os.path.relpath(file_path, repo_path))
            else:
                # Get the entire file content
                content = load_file(file_path)
            
            # Skip empty or binary files
            if not content:
                continue
            
            # Chunk the text and create embeddings
            chunks = chunk_text(content, chunk_size, overlap)
            chunk_embeddings = create_embeddings(chunks, model)
            
            if chunk_embeddings.size > 0:
                # Calculate file embedding (average of chunk embeddings)
                file_embedding = calculate_file_embedding(chunk_embeddings)
                
                if file_embedding is not None:
                    file_embeddings.append(file_embedding)
                    file_contents.append(content)
                    valid_files.append(file_path)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    if not valid_files:
        print(f"No valid files found for analysis in {analysis_type}.")
        return None, []
    
    print(f"Successfully processed {len(valid_files)} files for {analysis_type}.")
    
    # Convert to numpy array for efficient processing
    file_embeddings = np.array(file_embeddings)
    
    # Calculate similarity matrix between files
    similarity_matrix = cosine_similarity(file_embeddings)
    np.fill_diagonal(similarity_matrix, 1.0)  # Ensure self-similarity is exactly 1.0
    
    # Create visualization
    visualize_file_similarities(valid_files, file_embeddings, similarity_matrix, current_output_dir)
    
    # Save similarity matrix
    with open(os.path.join(current_output_dir, 'similarity_scores.txt'), 'w') as f:
        f.write("File Similarity Scores:\n")
        f.write("=====================\n\n")
        
        # Write header row with file names
        f.write("File,")
        f.write(",".join(os.path.basename(file) for file in valid_files))
        f.write("\n")
        
        # Write matrix rows
        for i, file in enumerate(valid_files):
            f.write(f"{os.path.basename(file)},")
            f.write(",".join(f"{similarity_matrix[i, j]:.4f}" for j in range(len(valid_files))))
            f.write("\n")
    
    # Generate recommendations
    with open(os.path.join(current_output_dir, 'recommendations.txt'), 'w') as f:
        f.write("File Relationship Recommendations:\n")
        f.write("===============================\n\n")
        
        # For each file, find the top 3 most similar files
        for i, file in enumerate(valid_files):
            similarities = [(j, similarity_matrix[i, j]) for j in range(len(valid_files)) if i != j]
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            f.write(f"File: {file}\n")
            f.write("Most similar files:\n")
            
            for j, sim in similarities[:3]:
                f.write(f"  - {valid_files[j]} (similarity: {sim:.4f})\n")
            
            f.write("\n")
    
    # Generate PR groupings based on similarity thresholds
    for threshold in [0.5, 0.6, 0.7, 0.8]:
        pr_groups = generate_pr_groups(valid_files, similarity_matrix, threshold)
        
        # Save PR groupings to a file
        with open(os.path.join(current_output_dir, f'pr_groups_threshold_{threshold:.1f}.txt'), 'w') as f:
            f.write(f"Suggested PR Groupings (Similarity Threshold {threshold:.1f}):\n")
            f.write("=" * 50 + "\n\n")
            
            for i, group in enumerate(pr_groups):
                if len(group) >= 1:  # Only include groups with at least one file
                    f.write(f"PR {i+1} ({len(group)} files):\n")
                    
                    # Calculate the average intra-group similarity
                    if len(group) > 1:
                        total_sim = 0
                        count = 0
                        for idx1, (_, node_id1) in enumerate(group):
                            for idx2, (_, node_id2) in enumerate(group[idx1+1:], idx1+1):
                                total_sim += similarity_matrix[node_id1, node_id2]
                                count += 1
                        avg_sim = total_sim / count if count > 0 else 0
                        f.write(f"  Average intra-group similarity: {avg_sim:.4f}\n")
                    
                    # List files in the group
                    for file_path, _ in group:
                        f.write(f"  - {file_path}\n")
                    f.write("\n")
        
        # Create visualization of PR groups
        visualize_pr_groups(valid_files, similarity_matrix, pr_groups, 
                           os.path.join(current_output_dir, f'pr_groups_threshold_{threshold:.1f}.png'))
    
    pr_groups_enhanced = generate_enhanced_pr_groups(valid_files, similarity_matrix, repo_path, threshold=0.6)

    # Save enhanced PR groupings
    with open(os.path.join(current_output_dir, 'pr_groups_enhanced.txt'), 'w') as f:
        f.write("Suggested PR Groupings (Enhanced with Directory & Import Analysis):\n")
        f.write("=" * 60 + "\n\n")
        
        for i, group in enumerate(pr_groups_enhanced):
            if len(group) >= 1:  # Only include groups with at least one file
                f.write(f"PR {i+1} ({len(group)} files):\n")
                
                # Calculate the average intra-group similarity
                if len(group) > 1:
                    total_sim = 0
                    count = 0
                    for idx1, (_, node_id1) in enumerate(group):
                        for idx2, (_, node_id2) in enumerate(group[idx1+1:], idx1+1):
                            total_sim += similarity_matrix[node_id1, node_id2]
                            count += 1
                    avg_sim = total_sim / count if count > 0 else 0
                    f.write(f"  Average intra-group similarity: {avg_sim:.4f}\n")
                
                # List files in the group
                for file_path, _ in group:
                    f.write(f"  - {file_path}\n")
                f.write("\n")
    
    # Create visualization of enhanced PR groups
    visualize_pr_groups(valid_files, similarity_matrix, pr_groups_enhanced, 
                       os.path.join(current_output_dir, 'pr_groups_enhanced.png'))
    
    # ADD NEW CODE HERE: Generate hierarchical PR groups
    # Try different max_distance values
    for max_distance in [0.3, 0.4, 0.5, 0.6]:
        pr_groups_hierarchical = generate_hierarchical_pr_groups(valid_files, similarity_matrix, max_distance)
        
        # Save hierarchical PR groupings
        with open(os.path.join(current_output_dir, f'pr_groups_hierarchical_{max_distance:.1f}.txt'), 'w') as f:
            f.write(f"Suggested PR Groupings (Hierarchical Clustering, Max Distance {max_distance:.1f}):\n")
            f.write("=" * 60 + "\n\n")
            
            for i, group in enumerate(pr_groups_hierarchical):
                if len(group) >= 1:  # Only include groups with at least one file
                    f.write(f"PR {i+1} ({len(group)} files):\n")
                    
                    # Calculate the average intra-group similarity
                    if len(group) > 1:
                        total_sim = 0
                        count = 0
                        for idx1, (_, node_id1) in enumerate(group):
                            for idx2, (_, node_id2) in enumerate(group[idx1+1:], idx1+1):
                                total_sim += similarity_matrix[node_id1, node_id2]
                                count += 1
                        avg_sim = total_sim / count if count > 0 else 0
                        f.write(f"  Average intra-group similarity: {avg_sim:.4f}\n")
                    
                    # List files in the group
                    for file_path, _ in group:
                        f.write(f"  - {file_path}\n")
                    f.write("\n")
        
        # Create visualization of hierarchical PR groups
        visualize_pr_groups(valid_files, similarity_matrix, pr_groups_hierarchical, 
                           os.path.join(current_output_dir, f'pr_groups_hierarchical_{max_distance:.1f}.png'))
    
    # Create dendrogram visualization
    create_dendrogram(valid_files, similarity_matrix, current_output_dir)
    
    return similarity_matrix, valid_files

def compare_analyses(repo_path, similarities_and_files, output_dir):
    """Create a comparative report of the three analysis approaches."""
    
    # Create report directory
    comparison_dir = os.path.join(output_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Create comparison report
    with open(os.path.join(comparison_dir, 'comparison_report.md'), 'w') as f:
        f.write("# Repository Analysis Comparison Report\n\n")
        f.write(f"Repository: `{repo_path}`\n\n")
        f.write("This report compares the results of analyzing the repository using three different approaches:\n\n")
        f.write("1. **All Files**: Analyzing all tracked files in the repository\n")
        f.write("2. **Changed Files (Full)**: Analyzing only files that have been modified but not yet committed, using their full content\n")
        f.write("3. **Changed Files (Diff Only)**: Analyzing only the specific changes (diffs) in modified files\n\n")
        
        f.write("## Analysis Statistics\n\n")
        f.write("| Analysis Type | Files Analyzed | Average Similarity | Most Similar Files |\n")
        f.write("|--------------|----------------|-------------------|-------------------|\n")
        
        # Add statistics for each analysis
        for analysis_type, (similarity_matrix, files) in similarities_and_files.items():
            if similarity_matrix is not None and len(files) > 0:
                # Calculate average similarity (excluding self-similarity on diagonal)
                total_sim = 0
                count = 0
                for i in range(len(similarity_matrix)):
                    for j in range(len(similarity_matrix)):
                        if i != j:
                            total_sim += similarity_matrix[i, j]
                            count += 1
                
                avg_sim = total_sim / count if count > 0 else 0
                
                # Find most similar file pair
                most_similar = (0, 0, 0)  # (i, j, similarity)
                for i in range(len(similarity_matrix)):
                    for j in range(i+1, len(similarity_matrix)):
                        if similarity_matrix[i, j] > most_similar[2]:
                            most_similar = (i, j, similarity_matrix[i, j])
                
                if most_similar[2] > 0:
                    most_similar_str = f"{os.path.basename(files[most_similar[0]])} & {os.path.basename(files[most_similar[1]])} ({most_similar[2]:.4f})"
                else:
                    most_similar_str = "N/A"
                
                f.write(f"| {analysis_type} | {len(files)} | {avg_sim:.4f} | {most_similar_str} |\n")
            else:
                f.write(f"| {analysis_type} | 0 | N/A | N/A |\n")
        
        f.write("\n## File Overlap Between Analyses\n\n")
        
        # Compare which files are present in each analysis
        all_files = set()
        for _, files in similarities_and_files.values():
            all_files.update(files)
        
        if all_files:
            f.write("| File | All Files | Changed Files (Full) | Changed Files (Diff Only) |\n")
            f.write("|------|----------|----------------------|---------------------------|\n")
            
            for file in sorted(all_files):
                in_all = "✓" if file in (similarities_and_files.get("all_files", (None, []))[1]) else ""
                in_changed = "✓" if file in (similarities_and_files.get("changed_files_full", (None, []))[1]) else ""
                in_diff = "✓" if file in (similarities_and_files.get("changed_files_diff_only", (None, []))[1]) else ""
                
                f.write(f"| {os.path.basename(file)} | {in_all} | {in_changed} | {in_diff} |\n")
        else:
            f.write("No files were present in any analysis.\n")
        
        f.write("\n## Key Insights\n\n")
        
        # Compare average similarity metrics
        avg_similarities = {}
        for analysis_type, (similarity_matrix, files) in similarities_and_files.items():
            if similarity_matrix is not None and len(files) > 0:
                total_sim = 0
                count = 0
                for i in range(len(similarity_matrix)):
                    for j in range(len(similarity_matrix)):
                        if i != j:
                            total_sim += similarity_matrix[i, j]
                            count += 1
                
                avg_similarities[analysis_type] = total_sim / count if count > 0 else 0
        
        if avg_similarities:
            # Sort by average similarity
            sorted_avgs = sorted(avg_similarities.items(), key=lambda x: x[1], reverse=True)
            
            f.write("### Similarity Comparison\n\n")
            f.write(f"The analysis with the highest average similarity between files is **{sorted_avgs[0][0]}** ({sorted_avgs[0][1]:.4f}).\n\n")
            
            if len(sorted_avgs) > 1:
                f.write("Comparison of average similarities:\n\n")
                for analysis_type, avg_sim in sorted_avgs:
                    f.write(f"- **{analysis_type}**: {avg_sim:.4f}\n")
            
            f.write("\n### Interpretation\n\n")
            
            # Generate insights based on which analysis showed higher similarities
            if "changed_files_diff_only" in avg_similarities and "changed_files_full" in avg_similarities:
                if avg_similarities["changed_files_diff_only"] > avg_similarities["changed_files_full"]:
                    f.write("The changes (diffs) in modified files show stronger similarities than their full content. ")
                    f.write("This suggests that recent changes are more closely related to each other than the overall file structures.\n\n")
                else:
                    f.write("The full content of modified files shows stronger similarities than just the changes. ")
                    f.write("This suggests that while recent changes may differ, the files serve related functions in the codebase.\n\n")
            
            if "all_files" in avg_similarities and "changed_files_full" in avg_similarities:
                if avg_similarities["changed_files_full"] > avg_similarities["all_files"]:
                    f.write("The modified files show stronger similarities compared to the entire repository. ")
                    f.write("This suggests that recent changes are focused on a specific, related part of the codebase.\n\n")
                else:
                    f.write("The entire repository shows stronger overall similarities than just the modified files. ")
                    f.write("This suggests that the modified files span different functional areas of the codebase.\n\n")
            
            f.write("### Recommendations\n\n")
            f.write("Based on this analysis:\n\n")
            
            if "changed_files_diff_only" in avg_similarities and avg_similarities.get("changed_files_diff_only", 0) > 0.7:
                f.write("- The changes show high similarity, suggesting possible code duplication or related functionality changes. Consider refactoring to reduce duplication.\n")
            elif "changed_files_diff_only" in avg_similarities and avg_similarities.get("changed_files_diff_only", 0) < 0.3:
                f.write("- The changes show low similarity, suggesting they affect different functional areas. Consider whether these changes should be in separate commits/branches.\n")
            
            if "all_files" in avg_similarities and "changed_files_full" in avg_similarities:
                if avg_similarities["changed_files_full"] > avg_similarities["all_files"] * 1.5:
                    f.write("- The current changes are much more closely related than the repository average. This is good for focused feature development.\n")
                elif avg_similarities["changed_files_full"] < avg_similarities["all_files"] * 0.5:
                    f.write("- The current changes span diverse parts of the codebase. Consider whether this should be split into multiple focused changes.\n")
        
        else:
            f.write("Insufficient data to generate insights.\n")
        
        f.write("\n## PR Grouping Comparison\n\n")
        f.write("The tool has generated PR groupings for each analysis approach. Here's a comparison:\n\n")
        
        # Compare PR groupings at threshold 0.7
        f.write("### PR Groupings Comparison (Threshold 0.7)\n\n")
        
        for analysis_type in similarities_and_files.keys():
            if analysis_type in ["changed_files_full", "changed_files_diff_only"]:
                pr_groups_file = os.path.join(output_dir, analysis_type, "pr_groups_threshold_0.7.txt")
                if os.path.exists(pr_groups_file):
                    # Count number of PRs and total files
                    pr_count = 0
                    file_count = 0
                    min_files = float('inf')
                    max_files = 0
                    
                    with open(pr_groups_file, 'r') as pr_file:
                        content = pr_file.read()
                        pr_count = content.count("PR ")
                        
                        # Extract PR sizes
                        import re
                        pr_sizes = re.findall(r"PR \d+ \((\d+) files\)", content)
                        if pr_sizes:
                            file_count = sum(int(size) for size in pr_sizes)
                            min_files = min(int(size) for size in pr_sizes)
                            max_files = max(int(size) for size in pr_sizes)
                    
                    f.write(f"**{analysis_type}**:\n")
                    f.write(f"- Number of PRs: {pr_count}\n")
                    f.write(f"- Total files in PRs: {file_count}\n")
                    f.write(f"- Smallest PR: {min_files} files\n")
                    f.write(f"- Largest PR: {max_files} files\n")
                    f.write(f"- See full groupings in: `{os.path.relpath(pr_groups_file, os.path.dirname(output_dir))}`\n\n")
        
        f.write("\n## Conclusion\n\n")
        f.write("The different analysis approaches provide complementary views of the codebase:\n\n")
        f.write("- **All Files** analysis provides a comprehensive view of the entire codebase structure\n")
        f.write("- **Changed Files (Full)** analysis focuses on the files currently being modified\n")
        f.write("- **Changed Files (Diff Only)** analysis focuses specifically on the current changes\n\n")
        f.write("By comparing these different views, you can better understand how your current changes relate to each other and to the overall codebase structure.\n\n")
        
        f.write("The PR groupings generated from each approach offer different organization strategies:\n\n")
        f.write("- **Changed Files (Full)** groupings organize files based on their overall functional similarity\n")
        f.write("- **Changed Files (Diff Only)** groupings organize files based on similarity of the actual changes\n\n")
        f.write("Consider the PR groupings from both approaches when organizing your changes for review. The diff-only analysis might be better for identifying files with similar changes, while the full content analysis might better reflect the codebase's logical organization.\n")
    
    print(f"Comparison report generated in {comparison_dir}/comparison_report.md")
    
    # Create comparative visualization if we have at least two analyses with data
    valid_analyses = 0
    for similarity_matrix, files in similarities_and_files.values():
        if similarity_matrix is not None and len(files) > 0:
            valid_analyses += 1
    
    if valid_analyses >= 2:
        create_comparison_visualization(similarities_and_files, comparison_dir)
        
        # Compare PR groupings visually
        compare_pr_groupings(similarities_and_files, output_dir, comparison_dir)

def compare_pr_groupings(similarities_and_files, output_dir, comparison_dir):
    """Create visualizations comparing PR groupings from different analyses."""
    # Focus on changed_files_full and changed_files_diff_only
    analysis_types = ["changed_files_full", "changed_files_diff_only"]
    thresholds = [0.7]  # Use 0.7 as the comparison threshold
    
    for threshold in thresholds:
        # Load PR groups data
        pr_groups_data = {}
        
        for analysis_type in analysis_types:
            if analysis_type in similarities_and_files:
                pr_groups_file = os.path.join(output_dir, analysis_type, f"pr_groups_threshold_{threshold:.1f}.txt")
                
                if os.path.exists(pr_groups_file):
                    # Parse PR groups
                    pr_groups = []
                    current_group = []
                    
                    with open(pr_groups_file, 'r') as f:
                        for line in f:
                            if line.strip().startswith("PR "):
                                if current_group:
                                    pr_groups.append(current_group)
                                current_group = []
                            elif line.strip().startswith("- "):
                                file_path = line.strip()[2:].strip()
                                current_group.append(file_path)
                    
                    if current_group:
                        pr_groups.append(current_group)
                    
                    pr_groups_data[analysis_type] = pr_groups
        
        # Create comparison visualization
        if len(pr_groups_data) >= 2:
            plt.figure(figsize=(12, 8))
            
            # Create grouped bar chart
            analysis_labels = [a.replace("_", " ").title() for a in pr_groups_data.keys()]
            
            # Calculate group sizes
            group_sizes = []
            for analysis_type in pr_groups_data.keys():
                sizes = [len(g) for g in pr_groups_data[analysis_type]]
                sizes.sort(reverse=True)
                group_sizes.append(sizes)
            
            # Plot group sizes
            max_groups = max(len(sizes) for sizes in group_sizes)
            x = np.arange(max_groups)
            width = 0.35
            
            for i, (analysis_type, sizes) in enumerate(zip(analysis_labels, group_sizes)):
                # Pad sizes with zeros if needed
                sizes_padded = sizes + [0] * (max_groups - len(sizes))
                plt.bar(x + (i - 0.5) * width, sizes_padded, width, label=analysis_type)
            
            plt.xlabel('PR Group Number')
            plt.ylabel('Number of Files')
            plt.title(f'PR Group Sizes Comparison (Threshold {threshold:.1f})')
            plt.xticks(x, [f'PR {i+1}' for i in range(max_groups)])
            plt.legend()
            
            plt.savefig(os.path.join(comparison_dir, f"pr_group_sizes_comparison_{threshold:.1f}.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create file overlap analysis
            file_sets = {}
            for analysis_type in pr_groups_data.keys():
                # Flatten all files
                all_files = set()
                for group in pr_groups_data[analysis_type]:
                    for file_path in group:
                        all_files.add(os.path.basename(file_path))
                file_sets[analysis_type] = all_files
            
            # Calculate Jaccard similarity between file sets
            if len(file_sets) >= 2:
                analysis_pairs = list(zip(file_sets.keys(), file_sets.values()))
                
                for i in range(len(analysis_pairs)):
                    for j in range(i+1, len(analysis_pairs)):
                        a_type1, files1 = analysis_pairs[i]
                        a_type2, files2 = analysis_pairs[j]
                        
                        # Calculate Jaccard similarity
                        intersection = len(files1.intersection(files2))
                        union = len(files1.union(files2))
                        similarity = intersection / union if union > 0 else 0
                        
                        print(f"File overlap between {a_type1} and {a_type2}: {similarity:.2f} ({intersection}/{union} files)")
            
            # Create a Venn diagram of PR groups
            try:
                from matplotlib_venn import venn2
                
                plt.figure(figsize=(10, 8))
                
                # Get the file sets for two analyses
                sets = [file_sets[a_type] for a_type in analysis_types]
                labels = [a_type.replace("_", " ").title() for a_type in analysis_types]
                
                venn2(sets, set_labels=labels)
                plt.title(f'File Overlap in PR Groups (Threshold {threshold:.1f})')
                plt.savefig(os.path.join(comparison_dir, f"pr_file_overlap_{threshold:.1f}.png"), dpi=300, bbox_inches='tight')
                plt.close()
                
            except ImportError:
                print("matplotlib-venn not installed. Skipping Venn diagram visualization.")

def create_comparison_visualization(similarities_and_files, output_dir):
    """Create visualizations comparing the different analyses."""
    
    # Create Venn diagram of file overlap
    try:
        from matplotlib_venn import venn3
        
        # Extract file basenames for each analysis
        file_sets = {}
        for analysis_type, (_, files) in similarities_and_files.items():
            if files:
                file_sets[analysis_type] = set(os.path.basename(f) for f in files)
        
        if len(file_sets) >= 2:
            plt.figure(figsize=(10, 8))
            
            if len(file_sets) == 3 and all(key in file_sets for key in ["all_files", "changed_files_full", "changed_files_diff_only"]):
                # Create a 3-way Venn diagram
                venn3([file_sets["all_files"], 
                       file_sets["changed_files_full"], 
                       file_sets["changed_files_diff_only"]],
                      set_labels=("All Files", "Changed Files (Full)", "Changed Files (Diff Only)"))
            elif len(file_sets) == 2:
                # Determine which two analyses we have
                keys = list(file_sets.keys())
                from matplotlib_venn import venn2
                venn2([file_sets[keys[0]], file_sets[keys[1]]], 
                      set_labels=(keys[0].replace("_", " ").title(), keys[1].replace("_", " ").title()))
            
            plt.title("File Overlap Between Analyses")
            plt.savefig(os.path.join(output_dir, "file_overlap_venn.png"), dpi=300, bbox_inches="tight")
            plt.close()
            
    except ImportError:
        print("matplotlib-venn not installed. Skipping Venn diagram visualization.")
    
    # Create bar chart of average similarities
    avg_similarities = {}
    for analysis_type, (similarity_matrix, files) in similarities_and_files.items():
        if similarity_matrix is not None and len(files) > 0:
            # Calculate average similarity (excluding self-similarity on diagonal)
            total_sim = 0
            count = 0
            for i in range(len(similarity_matrix)):
                for j in range(len(similarity_matrix)):
                    if i != j:
                        total_sim += similarity_matrix[i, j]
                        count += 1
            
            avg_similarities[analysis_type] = total_sim / count if count > 0 else 0
    
    if avg_similarities:
        plt.figure(figsize=(10, 6))
        analyses = list(avg_similarities.keys())
        avg_sims = [avg_similarities[a] for a in analyses]
        
        # Create readable labels
        labels = [a.replace("_", " ").title() for a in analyses]
        
        plt.bar(labels, avg_sims)
        plt.ylabel("Average Similarity Score")
        plt.title("Comparison of Average Similarity Scores")
        plt.ylim(0, max(avg_sims) * 1.2)  # Add some headroom
        
        # Add value labels on top of bars
        for i, v in enumerate(avg_sims):
            plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
        
        plt.savefig(os.path.join(output_dir, "avg_similarity_comparison.png"), dpi=300, bbox_inches="tight")
        plt.close()

def visualize_pr_groups(valid_files, similarity_matrix, pr_groups, output_file):
    """Create visualization of PR groups."""
    # Create a graph representation
    G = nx.Graph()
    
    # Add nodes
    for i, file_path in enumerate(valid_files):
        file_name = os.path.basename(file_path)
        G.add_node(i, name=file_name, path=file_path)
    
    # Add edges for all similarities
    for i in range(len(valid_files)):
        for j in range(i+1, len(valid_files)):
            similarity = similarity_matrix[i, j]
            if similarity > 0.3:  # Use a lower threshold for visualization
                G.add_edge(i, j, weight=similarity)
    
    # Assign colors to PR groups
    colors = list(mcolors.TABLEAU_COLORS.values())
    node_colors = ['#cccccc'] * len(valid_files)  # Default gray
    
    for group_idx, group in enumerate(pr_groups):
        color = colors[group_idx % len(colors)]
        for _, node_id in group:
            node_colors[node_id] = color
    
    # Position nodes using force-directed layout
    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(G, seed=42, k=0.3)
    
    # Draw nodes with PR group colors
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color=node_colors, alpha=0.8)
    
    # Draw edges with width based on similarity
    for u, v, data in G.edges(data=True):
        width = data['weight'] * 2
        alpha = min(0.8, max(0.1, data['weight'] - 0.3))
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, alpha=alpha)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, labels={i: G.nodes[i]['name'] for i in G.nodes}, font_size=8)
    
    # Add legend for PR groups
    legend_elements = []
    for i, group in enumerate(pr_groups):
        if group:
            color = colors[i % len(colors)]
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                          markersize=10, label=f'PR {i+1} ({len(group)} files)'))
    
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title('PR Groups Visualization')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_file_similarities(file_names, embeddings, similarity_matrix, output_dir):
    """Create visualizations of file similarities."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Create a t-SNE plot for dimensionality reduction
    if len(embeddings) > 1:  # Need at least 2 samples for t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        plt.figure(figsize=(12, 10))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
        
        # Add file names as labels
        for i, file_name in enumerate(file_names):
            plt.annotate(os.path.basename(file_name), 
                        (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                        fontsize=8)
        
        plt.title('t-SNE Visualization of File Similarities')
        plt.savefig(os.path.join(output_dir, 'tsne_visualization.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Try to identify clusters with K-means
        if len(embeddings) >= 3:  # Need at least 3 samples for meaningful clustering
            # Determine number of clusters - simple heuristic
            max_clusters = min(10, len(embeddings) // 2)
            inertias = []
            
            for k in range(1, max_clusters + 1):
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(embeddings)
                inertias.append(kmeans.inertia_)
            
            # Find elbow point (if available)
            k = 3  # Default
            if len(inertias) > 3:
                # Simple elbow detection
                diffs = np.diff(inertias)
                second_diffs = np.diff(diffs)
                k = np.argmax(second_diffs) + 2
                k = min(k, max_clusters)
            
            # Apply K-means with selected k
            kmeans = KMeans(n_clusters=k, random_state=42)
            clusters = kmeans.fit_predict(embeddings)
            
            # Plot with cluster colors
            plt.figure(figsize=(12, 10))
            scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                        c=clusters, cmap='viridis', alpha=0.7)
            
            # Add file names as labels
            for i, file_name in enumerate(file_names):
                plt.annotate(os.path.basename(file_name), 
                            (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                            fontsize=8)
            
            plt.title(f'File Clusters (K-means, k={k})')
            plt.colorbar(scatter, label='Cluster')
            plt.savefig(os.path.join(output_dir, 'file_clusters.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save cluster assignments to a file
            with open(os.path.join(output_dir, 'cluster_assignments.txt'), 'w') as f:
                f.write("File Cluster Assignments:\n")
                f.write("========================\n\n")
                
                for cluster_id in range(k):
                    f.write(f"Cluster {cluster_id}:\n")
                    cluster_files = [file_names[i] for i in range(len(file_names)) if clusters[i] == cluster_id]
                    for file in cluster_files:
                        f.write(f"  - {file}\n")
                    f.write("\n")
    
    # 3. Create a network graph visualization
    if len(file_names) > 1:
        plt.figure(figsize=(14, 12))
        G = nx.Graph()
        
        # Add nodes
        for i, file in enumerate(file_names):
            G.add_node(i, name=os.path.basename(file))
        
        # Add edges with similarity weight (only if similarity > threshold)
        threshold = 0.5  # Minimum similarity to draw an edge
        for i in range(len(file_names)):
            for j in range(i+1, len(file_names)):
                similarity = similarity_matrix[i, j]
                if similarity > threshold:
                    G.add_edge(i, j, weight=similarity)
        
        # Position nodes using force-directed layout
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=100, alpha=0.7)
        
        # Draw edges with width based on similarity
        for u, v, data in G.edges(data=True):
            width = data['weight'] * 3  # Scale width by similarity
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, alpha=0.5)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, labels={i: G.nodes[i]['name'] for i in G.nodes}, font_size=8)
        
        plt.title('File Similarity Network')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'similarity_network.png'), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    print("in main...")
    parser = argparse.ArgumentParser(description='Analyze file similarities in a Git repository using embeddings.')
    parser.add_argument('repo_path', help='Path to the Git repository')
    parser.add_argument('--model', default='all-mpnet-base-v2', help='Name of the sentence-transformers model to use')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Size of text chunks in words')
    parser.add_argument('--overlap', type=int, default=200, help='Overlap between chunks in words')
    parser.add_argument('--extensions', help='Comma-separated list of file extensions to analyze (e.g., ".py,.js,.html")')
    parser.add_argument('--diff-only', action='store_true', help='Analyze only changes (diffs) in files instead of entire files')
    parser.add_argument('--changed-only', action='store_true', help='Process only files that have been modified but not yet committed')
    parser.add_argument('--output-dir', default='similarity_analysis', help='Directory to save output files')
    parser.add_argument('--compare-all', action='store_true', help='Run all three analyses (all files, changed files, diff only) and compare results')
    parser.add_argument('--threshold', type=float, default=0.7, help='Similarity threshold for PR grouping (default: 0.7)')
    
    args = parser.parse_args()
    print("Going to start execution...")
    try:
        # Load model
        print(f"Loading model: {args.model}")
        model = SentenceTransformer(args.model)
        
        if args.compare_all:
            print("\nPerforming comparative analysis of all three approaches...")
            
            # Dictionary to store results from all three analyses
            all_results = {}
            
            # Run analysis 1: All files
            sim_matrix, files = analyze_repo(
                args.repo_path, model, args.chunk_size, args.overlap, 
                args.extensions, False, False, args.output_dir
            )
            all_results["all_files"] = (sim_matrix, files)
            
            # Run analysis 2: Changed files (full content)
            sim_matrix, files = analyze_repo(
                args.repo_path, model, args.chunk_size, args.overlap, 
                args.extensions, True, False, args.output_dir
            )
            all_results["changed_files_full"] = (sim_matrix, files)

            # Run analysis 3: Changed files (diff only)
            sim_matrix, files = analyze_repo(
                args.repo_path, model, args.chunk_size, args.overlap, 
                args.extensions, True, True, args.output_dir
            )
            all_results["changed_files_diff_only"] = (sim_matrix, files)
            
            # Generate comparison report
            compare_analyses(args.repo_path, all_results, args.output_dir)
            
            print(f"\nComparative analysis complete. Results saved to {args.output_dir}/")
            
        else:
            # Get files from repository
            repo_files = get_repo_files(args.repo_path, args.extensions, args.changed_only)
            if not repo_files:
                print("No files found in the repository matching the criteria.")
                return
            
            print(f"Found {len(repo_files)} files to analyze.")
            
            # Process each file
            file_embeddings = []
            file_contents = []
            valid_files = []
            
            for file_path in tqdm(repo_files, desc="Processing files"):
                try:
                    if args.diff_only:
                        # Get only the diff
                        content = get_git_diff(args.repo_path, os.path.relpath(file_path, args.repo_path))
                    else:
                        # Get the entire file content
                        content = load_file(file_path)
                    
                    # Skip empty or binary files
                    if not content:
                        continue
                    
                    # Chunk the text and create embeddings
                    chunks = chunk_text(content, args.chunk_size, args.overlap)
                    chunk_embeddings = create_embeddings(chunks, model)
                    
                    if chunk_embeddings.size > 0:
                        # Calculate file embedding (average of chunk embeddings)
                        file_embedding = calculate_file_embedding(chunk_embeddings)
                        
                        if file_embedding is not None:
                            file_embeddings.append(file_embedding)
                            file_contents.append(content)
                            valid_files.append(file_path)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
            
            if not valid_files:
                print("No valid files found for analysis.")
                return
            
            print(f"Successfully processed {len(valid_files)} files.")
            
            # Convert to numpy array for efficient processing
            file_embeddings = np.array(file_embeddings)
            
            # Calculate similarity matrix between files
            similarity_matrix = cosine_similarity(file_embeddings)
            
            # Create visualization
            visualize_file_similarities(valid_files, file_embeddings, similarity_matrix, args.output_dir)
            
            # Save similarity matrix
            with open(os.path.join(args.output_dir, 'similarity_scores.txt'), 'w') as f:
                f.write("File Similarity Scores:\n")
                f.write("=====================\n\n")
                
                # Write header row with file names
                f.write("File,")
                f.write(",".join(os.path.basename(file) for file in valid_files))
                f.write("\n")
                
                # Write matrix rows
                for i, file in enumerate(valid_files):
                    f.write(f"{os.path.basename(file)},")
                    f.write(",".join(f"{similarity_matrix[i, j]:.4f}" for j in range(len(valid_files))))
                    f.write("\n")
            
            # Generate recommendations
            with open(os.path.join(args.output_dir, 'recommendations.txt'), 'w') as f:
                f.write("File Relationship Recommendations:\n")
                f.write("===============================\n\n")
                
                # For each file, find the top 3 most similar files
                for i, file in enumerate(valid_files):
                    similarities = [(j, similarity_matrix[i, j]) for j in range(len(valid_files)) if i != j]
                    similarities.sort(key=lambda x: x[1], reverse=True)
                    
                    f.write(f"File: {file}\n")
                    f.write("Most similar files:\n")
                    
                    for j, sim in similarities[:3]:
                        f.write(f"  - {valid_files[j]} (similarity: {sim:.4f})\n")
                    
                    f.write("\n")
            
            # Generate PR groups based on the specified threshold
            pr_groups = generate_pr_groups(valid_files, similarity_matrix, args.threshold)
            
            # Save PR groupings to a file
            with open(os.path.join(args.output_dir, f'pr_groups_threshold_{args.threshold:.1f}.txt'), 'w') as f:
                f.write(f"Suggested PR Groupings (Similarity Threshold {args.threshold:.1f}):\n")
                f.write("=" * 50 + "\n\n")
                
                for i, group in enumerate(pr_groups):
                    if len(group) >= 1:  # Only include groups with at least one file
                        f.write(f"PR {i+1} ({len(group)} files):\n")
                        
                        # Calculate the average intra-group similarity
                        if len(group) > 1:
                            total_sim = 0
                            count = 0
                            for idx1, (_, node_id1) in enumerate(group):
                                for idx2, (_, node_id2) in enumerate(group[idx1+1:], idx1+1):
                                    total_sim += similarity_matrix[node_id1, node_id2]
                                    count += 1
                            avg_sim = total_sim / count if count > 0 else 0
                            f.write(f"  Average intra-group similarity: {avg_sim:.4f}\n")
                        
                        # List files in the group
                        for file_path, _ in group:
                            f.write(f"  - {file_path}\n")
                        f.write("\n")
            
            # Create visualization of PR groups
            visualize_pr_groups(valid_files, similarity_matrix, pr_groups, 
                               os.path.join(args.output_dir, f'pr_groups_threshold_{args.threshold:.1f}.png'))
            
            print(f"Analysis complete. Results saved to {args.output_dir}/")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()