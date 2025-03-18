#!/usr/bin/env python3
"""
Dynamic PR Categorizer with Path Support
"""
import os
import subprocess
import re
import argparse
import math
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import Normalizer
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

def validate_git_repo(path):
    """Check if the path contains a valid git repository"""
    git_dir = os.path.join(path, '.git')
    if not os.path.isdir(git_dir):
        raise ValueError(f"Not a git repository: {path}")

def get_historical_cooccurrence(repo_path):
    """Analyze git log to build file relationships"""
    print(f"Analyzing git history in {repo_path}...")
    log = subprocess.check_output(
        ['git', 'log', '--numstat', '--pretty=format:'],
        cwd=repo_path
    ).decode(errors='ignore')
    
    cooccur = defaultdict(lambda: defaultdict(int))
    files = set()
    
    for commit in log.split('\n\n'):
        paths = set()
        for line in commit.split('\n'):
            if not line.strip():
                continue
            parts = re.split(r'\s+', line.strip(), 2)
            if len(parts) < 3:
                continue
            add, sub, path = parts
            if path and not path.startswith('.'):
                paths.add(path)
                files.add(path)
        
        for p1 in paths:
            for p2 in paths:
                if p1 != p2:
                    cooccur[p1][p2] += 1
                    cooccur[p2][p1] += 1
    
    return dict(cooccur), list(files)

def get_current_changes(repo_path):
    """Get list of modified files from git status"""
    output = subprocess.check_output(
        ['git', 'status', '--porcelain'],
        cwd=repo_path
    ).decode(errors='ignore')
    return [line[3:] for line in output.splitlines() if line.strip()]

def structural_features(paths):
    """Enhanced directory weighting with depth awareness"""
    features = []
    for path in paths:
        parts = re.split(r'[/_.-]', path)
        weighted = []
        for i, part in enumerate(parts):
            # Give more weight to deeper directories
            depth_weight = int(math.log(len(parts) - i + 1)) * 2
            # Special handling for file extensions
            if '.' in part and i == len(parts)-1:
                ext = part.split('.')[-1]
                weighted.append(f"ext_{ext}")
            weighted.append(f"{part}{'/'*depth_weight}")
        features.append(' '.join(weighted))
    return features

def tfidf_features(texts):
    """Extract important keywords from filenames"""
    vectorizer = TfidfVectorizer(
        stop_words=None,  # Disable stopwords for short technical terms
        token_pattern=r'(?u)\b[a-zA-Z]{3,}\b',
        ngram_range=(1, 2), # capture bigrams
        min_df=2 , # Consider even single-occurrence terms
        max_features=1000
    )
    try:
        return vectorizer.fit_transform(texts)
    except ValueError:
        # Fallback for pure numeric directories
        return np.zeros((len(texts), 1))

def cluster_files(paths, cooccur_matrix, all_files):
    """More granular clustering with adjusted parameters"""
    structural = tfidf_features(structural_features(paths))
    cooccur = np.array([
        [cooccur_matrix.get(p1, {}).get(p2, 0) for p2 in paths]
        for p1 in paths
    ])
    
    combined = np.hstack([
        Normalizer().fit_transform(structural.toarray()),
        Normalizer().fit_transform(cooccur) * 1.5  # Increase co-occurrence weight
    ])

    dynamic_min_cluster_size = max(2, len(paths) // 20) # e.g., 5% of total files
    clusterer = HDBSCAN(
        min_cluster_size=2,
        min_samples=1,
        cluster_selection_epsilon=0.5,
        metric='cosine'
    )
    labels = clusterer.fit_predict(combined)
    
    return labels

def generate_labels(clusters, paths):
    """More precise label creation with technical term prioritization"""
    labels = []
    for cluster_id, members in clusters.items():
        # Extract key components
        components = {
            'depth': [],
            'tech_terms': [],
            'file_types': set()
        }
        
        for path in members:
            parts = path.split('/')
            # Capture directory structure patterns
            components['depth'].append(len(parts))
            
            # Extract technical terms
            for part in parts:
                if re.match(r'^(utils|service|model|config|test)', part, re.I):
                    components['tech_terms'].append(part)
            
            # Track file extensions
            if '.' in path:
                components['file_types'].add(path.split('.')[-1])
        
        # Generate label components
        label_parts = []
        
        # 1. Prioritize technical terms
        unique_terms = list(set(components['tech_terms']))[:3]
        
        # 2. Add file types if significant
        if len(components['file_types']) > 0:
            label_parts.append('/'.join(sorted(components['file_types'])))
        
        # 3. Add depth indicator
        avg_depth = np.mean(components['depth'])
        if avg_depth > 3:
            label_parts.append("DeepModule")
        elif avg_depth > 2:
            label_parts.append("MidModule")
        
        # 4. Combine terms
        label = ' '.join(unique_terms + label_parts)
        
        # Clean up
        label = re.sub(r'[_/-]+', ' ', label).title()
        labels.append(label[:40])  # Keep labels concise
    
    return labels

def secondary_cluster(files, cooccur_matrix):
    """Further cluster large groups into finer clusters"""
    structural = tfidf_features(structural_features(files))
    cooccur = np.array([
        [cooccur_matrix.get(p1, {}).get(p2, 0) for p2 in files]
        for p1 in files
    ])

    combined = np.hstack([
        Normalizer().fit_transform(structural.toarray()),
        Normalizer().fit_transform(cooccur) * 1.5
    ])

    clusterer = HDBSCAN(
        min_cluster_size=2,
        min_samples=1,
        cluster_selection_epsilon=0.3,  # Finer clustering
        metric='cosine'
    )
    labels = clusterer.fit_predict(combined)
    return labels


def main():
    parser = argparse.ArgumentParser(
        description='Automatically group git changes into PR categories'
    )
    parser.add_argument('--path', 
                      type=str, 
                      default='.',
                      help='Path to git repository')
    args = parser.parse_args()
    
    # Validate paths
    repo_path = os.path.abspath(args.path)
    if not os.path.isdir(repo_path):
        raise FileNotFoundError(f"Directory not found: {repo_path}")
    validate_git_repo(repo_path)
    
    # Get data
    cooccur, all_files = get_historical_cooccurrence(repo_path)
    changed_files = get_current_changes(repo_path)
    
    if not changed_files:
        print("No changes detected!")
        return
    
    # Cluster files
    labels = cluster_files(changed_files, cooccur, all_files)
    
    # Group files by cluster
    clusters = defaultdict(list)
    for path, label in zip(changed_files, labels):
        clusters[label].append(path)
    
    # Split large clusters further
    final_clusters = defaultdict(list)
    new_label_id = max(clusters.keys(), default=-1) + 1

    for label_id, files in clusters.items():
        if len(files) > 30:  # Threshold for re-clustering
            sub_labels = secondary_cluster(files, cooccur)
            temp = defaultdict(list)
            for f, sub_label in zip(files, sub_labels):
                temp[sub_label].append(f)
            for sub_id, sub_files in temp.items():
                final_clusters[new_label_id] = sub_files
                new_label_id += 1
        else:
            final_clusters[label_id] = files
    
    # Generate labels
    # Flatten all final cluster files to use in label generation
    all_final_files = [file for files in final_clusters.values() for file in files]
    cluster_labels = generate_labels(final_clusters, all_final_files)
    
    # Print results
    print(f"\nAuto-generated PR Categories for {repo_path}:")
    for label, (cluster_id, files) in zip(cluster_labels, final_clusters.items()):
        print(f"\n=== {label} ===")
        print(f"Files ({len(files)}):")
        for f in files[:5]:
            print(f"  {f}")
        if len(files) > 5:
            print(f"  ...and {len(files)-5} more")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)