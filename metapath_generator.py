# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm
import random
import dgl


class MultiSourceAttentionFusion(nn.Module):
    """Multi-source attention fusion module"""

    def __init__(self, embedding_dim, hidden_dim, num_heads, dropout=0.1):
        super(MultiSourceAttentionFusion, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout

        # Attention networks
        self.attention_layer1 = nn.Linear(embedding_dim, hidden_dim)
        self.attention_layer2 = nn.Linear(hidden_dim, num_heads)
        self.dropout_layer = nn.Dropout(dropout)

        # Parameter initialization
        nn.init.xavier_normal_(self.attention_layer1.weight)
        nn.init.xavier_normal_(self.attention_layer2.weight)

    def forward(self, embeddings_list):
        """
        embeddings_list: List of tensors [num_embeddings, num_nodes, embedding_dim]
        Returns: fused_embedding [num_nodes, embedding_dim]
        """
        # Stack all embeddings [num_embeddings, num_nodes, embedding_dim]
        stacked_embeddings = torch.stack(embeddings_list, dim=0)
        num_embeddings, num_nodes, embedding_dim = stacked_embeddings.shape

        # Compute attention weights
        # [num_embeddings, num_nodes, embedding_dim] -> [num_embeddings, num_nodes, hidden_dim]
        attn_hidden = torch.tanh(self.attention_layer1(stacked_embeddings))
        attn_hidden = self.dropout_layer(attn_hidden)

        # [num_embeddings, num_nodes, hidden_dim] -> [num_embeddings, num_nodes, num_heads]
        attn_scores = self.attention_layer2(attn_hidden)

        # Apply softmax for each head [num_embeddings, num_nodes, num_heads]
        attn_weights = F.softmax(attn_scores, dim=0)

        # Multi-head attention fusion
        fused_embeddings = []
        for head in range(self.num_heads):
            # Get current head weights [num_embeddings, num_nodes, 1]
            head_weights = attn_weights[:, :, head:head + 1]
            # Weighted sum [num_nodes, embedding_dim]
            head_fused = torch.sum(head_weights * stacked_embeddings, dim=0)
            fused_embeddings.append(head_fused)

        # Average results from multiple heads
        final_fused = torch.mean(torch.stack(fused_embeddings, dim=0), dim=0)

        return final_fused


class RegulatoryPathSubgraphGenerator:
    def __init__(self, args, device='cuda'):
        self.device = device
        self.args = args
        self.metapaths = args.path_list
        self.walk_length = args.path_walk_length
        self.num_walks = args.path_num_walks

    def generate_integer_list(self, start, end):
        """Generate node ID list according to Q2 method"""
        return list(range(start, end))

    def metapath_random_walk(self, G, start_node, metapath, walk_length):
        """Random walk based on metapaths"""
        # Convert metapath to type sequence
        if metapath == 'DTP':
            actual_metapath = ['D', 'T', 'P']
        elif metapath == 'DTDP':
            actual_metapath = ['D', 'T', 'D', 'P']
        elif metapath == 'DTPTD':
            actual_metapath = ['D', 'T', 'P', 'T', 'D']
        elif metapath == 'drug-protein-disease':
            actual_metapath = ['D', 'P', 'T']
        elif metapath == 'drug-protein-protein-disease':
            actual_metapath = ['D', 'P', 'P', 'T']
        elif metapath == 'disease-protein-drug-protein':
            actual_metapath = ['T', 'P', 'D', 'P']
        elif metapath == 'drug-protein-drug':
            actual_metapath = ['D', 'P', 'D']
        elif metapath == 'drug-protein-protein-drug':
            actual_metapath = ['D', 'P', 'P', 'D']
        else:
            actual_metapath = list(metapath)

        walk = [start_node]
        cur_node = start_node
        cur_metapath_idx = 0

        while len(walk) < walk_length:
            # Get current and next node types
            cur_type = actual_metapath[cur_metapath_idx % len(actual_metapath)]
            next_type = actual_metapath[(cur_metapath_idx + 1) % len(actual_metapath)]

            # Get candidate neighboring nodes of corresponding type
            next_candidates = [n for n in G.neighbors(cur_node)
                               if G.nodes[n]['node_type'] == next_type]

            if not next_candidates:
                break

            # Randomly select a neighbor
            next_node = random.choice(next_candidates)
            walk.append(next_node)
            cur_node = next_node
            cur_metapath_idx += 1

        return walk

    def generate_subgraph(self, G, start_node, metapath, walk_length, num_walks):
        """Generate subgraph with deduplication"""
        subgraph_nodes = set()
        unique_walks = set()
        attempts = 0
        max_attempts = num_walks * 3

        while len(unique_walks) < num_walks and attempts < max_attempts:
            walk = self.metapath_random_walk(G, start_node, metapath, walk_length)
            walk_tuple = tuple(walk)

            if walk_tuple not in unique_walks:
                unique_walks.add(walk_tuple)
                subgraph_nodes.update(walk)

            attempts += 1

        return G.subgraph(subgraph_nodes)

    def subgraph_adj_matrix_with_original_shape(self, G, subgraph):
        """
        Generate subgraph adjacency matrix with original graph shape according to Q2 implementation
        This is key - preserve original dimension but become sparse, ensure node mapping is correct
        """
        n = len(G.nodes())
        subgraph_nodes = subgraph.nodes()
        full_adj_matrix = nx.adjacency_matrix(G).todense()
        subgraph_adj_matrix = np.zeros((n, n))

        for i in subgraph_nodes:
            for j in subgraph_nodes:
                subgraph_adj_matrix[i, j] = full_adj_matrix[i, j]

        return subgraph_adj_matrix

    def build_heterograph_from_train_data_q2_style(self, train_drdi_edges, d_data):
        """
        Build heterograph according to Q2 method - consecutive ID assignment
        Key: Node range based on full dataset, not training data, avoiding information leakage
        """
        # Get training edges
        drdi_edges = train_drdi_edges  # Current fold drug-disease edges
        drpr_edges = d_data['drpr']  # Full drug-protein edges
        dipr_edges = d_data['dipr']  # Full disease-protein edges

        # Create heterograph
        G = nx.Graph()

        # Q2 key: Node range based on full dataset, not training data only
        # Use full dataset node counts to define node range
        drug_count = d_data['drug_number']
        disease_count = d_data['disease_number']
        protein_count = d_data['protein_number']

        # Q2-style consecutive ID assignment:
        # Drug nodes: [0, drug_count-1]
        # Disease nodes: [drug_count, drug_count+disease_count-1]
        # Protein nodes: [drug_count+disease_count, drug_count+disease_count+protein_count-1]

        drug_node_list = self.generate_integer_list(0, drug_count)
        disease_node_list = self.generate_integer_list(drug_count, drug_count + disease_count)
        protein_node_list = self.generate_integer_list(drug_count + disease_count,
                                                       drug_count + disease_count + protein_count)

        # Add all nodes and their type labels
        G.add_nodes_from(drug_node_list, node_type="D")
        G.add_nodes_from(disease_node_list, node_type="T")  # T represents Target/Disease
        G.add_nodes_from(protein_node_list, node_type="P")

        # Store node range information
        self.drug_node_range = (0, drug_count)
        self.disease_node_range = (drug_count, drug_count + disease_count)
        self.protein_node_range = (drug_count + disease_count, drug_count + disease_count + protein_count)
        self.total_nodes = drug_count + disease_count + protein_count

        # Add edges (do not use training data to avoid information leakage)
        drdi_edges_remapped = drdi_edges.copy()
        drdi_edges_remapped[:, 1] += drug_count

        drpr_edges_remapped = drpr_edges.copy()
        drpr_edges_remapped[:, 1] += (drug_count + disease_count)

        dipr_edges_remapped = dipr_edges.copy()
        dipr_edges_remapped[:, 0] += drug_count
        dipr_edges_remapped[:, 1] += (drug_count + disease_count)

        all_edges_remapped = np.vstack([drdi_edges_remapped, drpr_edges_remapped, dipr_edges_remapped])
        all_edges_list = [(row[0], row[1]) for row in all_edges_remapped]

        G.add_edges_from(all_edges_list)

        print(f"Q2-style heterograph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print(
            f"Node ranges - Drug: {self.drug_node_range}, Disease: {self.disease_node_range}, Protein: {self.protein_node_range}")
        print(f"Total nodes: {self.total_nodes}")

        return G

    def generate_regulatory_path_subgraphs(self, train_drdi_edges, d_data):
        """
        Generate regulatory path subgraphs for HSGE processing only
        Key: Preserve original graph shape, ensure node mapping is correct
        Note: Only HSGE processing supported for metapaths (no SGE option)
        """
        print("=== Generating Regulatory Path Subgraphs for HSGE Processing ===")
        print("Note: Metapaths require heterogeneous processing, SGE not supported")

        # 1. Build heterograph from training data (Q2 method)
        G = self.build_heterograph_from_train_data_q2_style(train_drdi_edges, d_data)

        # 2. Generate sparse adjacency matrix for each regulatory path (preserve original shape)
        path_adj_matrices = {}

        for i, path in enumerate(self.metapaths):
            print(f"Processing regulatory path {i + 1}: {path}")

            drug_nodes = list(range(0, self.drug_node_range[1]))
            disease_nodes = list(range(self.disease_node_range[0], self.disease_node_range[1]))
            start_nodes = drug_nodes + disease_nodes
            node_type_name = "Drug+Disease"

            print(f"  Starting from {node_type_name} nodes only: {len(start_nodes)} nodes")

            # Regulatory path random walk from specific type nodes only
            subgraphs = {}
            for start_node in tqdm(start_nodes, desc=f"Generating {path} subgraphs from {node_type_name}",
                                   leave=False):
                subgraph = self.generate_subgraph(G, start_node, path, self.walk_length, self.num_walks)
                subgraphs[start_node] = subgraph

            # Merge all subgraphs (collect all nodes and edges visited)
            combined_subgraph = nx.Graph()
            for _, subgraph in subgraphs.items():
                combined_subgraph = nx.compose(combined_subgraph, subgraph)

            # Generate sparse adjacency matrix with original shape
            combined_subgraph_adj_matrix = self.subgraph_adj_matrix_with_original_shape(G, combined_subgraph)

            path_adj_matrices[path] = combined_subgraph_adj_matrix

            # Statistics
            total_nodes = combined_subgraph_adj_matrix.shape[0]
            total_edges = np.sum(combined_subgraph_adj_matrix) / 2  # Undirected graph
            isolated_nodes = np.sum(combined_subgraph_adj_matrix.sum(axis=1) == 0)
            connectivity_ratio = 1 - (isolated_nodes / total_nodes)

            print(f"Adjacency matrix for {path}: shape {combined_subgraph_adj_matrix.shape}")
            print(f"  Total edges: {int(total_edges)}, Isolated nodes: {isolated_nodes}/{total_nodes}")
            print(f"  Connectivity ratio: {connectivity_ratio:.3f}")
            print(f"  Note: Preserves original graph shape with sparse connections for HSGE processing")

        print("=== Regulatory Path Subgraph Generation Complete ===")
        print("All metapath subgraphs prepared for HSGE processing")
        return path_adj_matrices, G


def generate_fold_path_data(train_drdi_edges, d_data, args, device='cuda', fold_idx=0):
    """
    Generate regulatory path data for each fold, supporting caching
    Note: Only HSGE processing supported for metapaths

    Key characteristics:
    1. Path subgraph adjacency matrix preserves original heterograph's full shape [total_nodes, total_nodes]
    2. Visited nodes become sparse connected graph (others 0)
    3. Use original features instead of one-hot
    4. HSGE naturally supports processing heterogeneous graphs including sparse connections
    """
    import os
    import pickle
    import hashlib

    print(f"Generating regulatory path data for HSGE processing only")
    print("Note: Metapaths require heterogeneous processing capabilities")

    # Create cache directory
    cache_dir = os.path.join("cache", args.dataset, "path_data")
    os.makedirs(cache_dir, exist_ok=True)

    # Generate cache filename (based on key parameters hash)
    cache_key_data = {
        'fold_idx': fold_idx,
        'seed': args.seed,
        'negative_rate': args.negative_rate,
        'path_list': args.path_list,
        'path_walk_length': args.path_walk_length,
        'path_num_walks': args.path_num_walks,
        'processor': 'hsge',  # Fixed to HSGE for metapaths
        'unified_hetero_processor': args.unified_hetero_processor,
        'train_edges_hash': hashlib.md5(str(sorted(train_drdi_edges.tolist())).encode()).hexdigest()[:8]
    }

    cache_key_str = str(sorted(cache_key_data.items()))
    cache_hash = hashlib.md5(cache_key_str.encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f"fold_{fold_idx}_{cache_hash}.pkl")

    # Try loading from cache
    if os.path.exists(cache_file):
        print(f"Loading cached regulatory path data from: {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)

        print(f"Successfully loaded cached regulatory path data")
        print(f"  Cached paths: {list(cached_data['path_adj_matrices'].keys())}")
        print(f"  Total nodes: {cached_data['total_nodes']}")
        print(f"  Processor type: {cached_data['processor_type']}")

        return cached_data

    # Generate new regulatory path data
    print("Generating new regulatory path data for HSGE processing...")
    generator = RegulatoryPathSubgraphGenerator(args, device)

    # Generate regulatory path subgraph adjacency matrices (sparse, including isolated nodes)
    path_adj_matrices, heterograph = generator.generate_regulatory_path_subgraphs(train_drdi_edges, d_data)

    # Validation
    for path, adj_matrix in path_adj_matrices.items():
        assert adj_matrix.shape[0] == adj_matrix.shape[1], f"Error: {path} adjacency matrix not square"
        assert adj_matrix.shape[0] == generator.total_nodes, f"Error: {path} adjacency matrix shape mismatch"

    print("Validation passed: All path matrices have correct shape for HSGE processing")
    print(f"Using HSGE processor with {generator.total_nodes} nodes")
    print(f"Architecture mode: {'Unified' if args.unified_hetero_processor else 'Separated'}")

    # Prepare data for caching
    data_to_cache = {
        'path_adj_matrices': path_adj_matrices,
        'heterograph': heterograph,
        'ndls_features': None,  # No NDLS features
        'node_ranges': {
            'drug': generator.drug_node_range,
            'disease': generator.disease_node_range,
            'protein': generator.protein_node_range
        },
        'total_nodes': generator.total_nodes,
        'generator': generator,  # Store generator instance for later use
        'processor_type': 'hsge',  # Fixed to HSGE for metapaths
        'unified_hetero_processor': args.unified_hetero_processor,
        'cache_info': {
            'generated_time': str(pd.Timestamp.now()),
            'fold_idx': fold_idx,
            'cache_key': cache_key_data
        }
    }

    # Save cache
    print(f"Saving regulatory path data to cache: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(data_to_cache, f)
    print(f"Successfully cached regulatory path data")

    return data_to_cache