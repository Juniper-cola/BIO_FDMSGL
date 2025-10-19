# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import dgl.nn.pytorch
from parse_args import args
from spectral_sge import SpectralSGE, SpectralHSGE
from metapath_generator import MultiSourceAttentionFusion
import torch.nn.functional as F
import numpy as np
import dgl
import scipy.sparse as sp

device = torch.device('cuda')


def multiple_operator(a, b):
    return a * b


def rotate_operator(a, b):
    a_re, a_im = a.chunk(2, dim=-1)
    b_re, b_im = b.chunk(2, dim=-1)
    message_re = a_re * b_re - a_im * b_im
    message_im = a_re * b_im + a_im * b_re
    message = torch.cat([message_re, message_im], dim=-1)
    return message


class UnifiedFeatureExtractor:
    """
    Unified feature extraction utility, supports MLP classifier
    """

    def __init__(self):
        pass

    def extract_pair_features(self, drug_embeddings, disease_embeddings, pairs, feature_type='all'):
        """
        Unified pair feature extraction interface

        Args:
            drug_embeddings: [num_drugs, embedding_dim]
            disease_embeddings: [num_diseases, embedding_dim]
            pairs: [num_pairs, 2] - (drug_idx, disease_idx)
            feature_type: 'concat', 'element_wise', 'separate', 'all'

        Returns:
            feature_dict: Dictionary containing different types of features
        """
        drug_feats = drug_embeddings[pairs[:, 0]]  # [num_pairs, embedding_dim]
        disease_feats = disease_embeddings[pairs[:, 1]]  # [num_pairs, embedding_dim]

        feature_dict = {}

        if feature_type in ['concat', 'all']:
            feature_dict['concat'] = torch.cat([drug_feats, disease_feats], dim=1)

        if feature_type in ['element_wise', 'all']:
            feature_dict['element_wise'] = drug_feats * disease_feats

        if feature_type in ['hadamard', 'all']:
            feature_dict['hadamard'] = drug_feats * disease_feats

        if feature_type in ['l1_distance', 'all']:
            feature_dict['l1_distance'] = torch.abs(drug_feats - disease_feats)

        if feature_type in ['l2_distance', 'all']:
            feature_dict['l2_distance'] = (drug_feats - disease_feats) ** 2

        if feature_type in ['cosine_sim', 'all']:
            drug_norm = F.normalize(drug_feats, p=2, dim=1)
            disease_norm = F.normalize(disease_feats, p=2, dim=1)
            feature_dict['cosine_sim'] = torch.sum(drug_norm * disease_norm, dim=1, keepdim=True)

        if feature_type in ['separate', 'all']:
            feature_dict['separate'] = (drug_feats, disease_feats)

        if feature_type in ['advanced', 'all']:
            concat_feats = torch.cat([drug_feats, disease_feats], dim=1)
            element_wise = drug_feats * disease_feats
            l1_dist = torch.abs(drug_feats - disease_feats)
            feature_dict['advanced'] = torch.cat([concat_feats, element_wise, l1_dist], dim=1)

        return feature_dict


class MLPClassifier(nn.Module):
    """
    MLP classifier - maintains compatibility with existing architectures
    """

    def __init__(self, input_dim, hidden_dims=[1024, 1024, 256], num_classes=2, dropout=0.4,
                 feature_combination='element_wise'):
        super(MLPClassifier, self).__init__()
        self.feature_combination = feature_combination

        if feature_combination == 'concat':
            actual_input_dim = input_dim * 2
        elif feature_combination == 'advanced':
            actual_input_dim = input_dim * 4  # concat + element_wise + l1_distance
        else:
            actual_input_dim = input_dim

        layers = []
        prev_dim = actual_input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.mlp = nn.Sequential(*layers)

    def forward(self, feature_dict):
        """
        Forward propagation

        Args:
            feature_dict: Feature dictionary
        Returns:
            logits: [batch_size, num_classes]
        """
        if self.feature_combination in feature_dict:
            x = feature_dict[self.feature_combination]
        else:
            x = feature_dict['element_wise']

        return self.mlp(x)


class UnifiedEmbeddingModule(nn.Module):
    """
    Unified embedding module supporting two HSGE architecture modes:
    Mode A: Unified HSGE for both heterograph and metapaths
    Mode B: Separate HSGE for heterograph, shared HSGE for all metapaths
    """

    def __init__(self, args, meta_g, path_names, total_nodes, output_dim, drug_number, disease_number):
        super(UnifiedEmbeddingModule, self).__init__()

        self.args = args
        self.meta_g = meta_g
        self.path_names = path_names
        self.output_dim = output_dim
        self.total_nodes = total_nodes
        self.drug_number = drug_number
        self.disease_number = disease_number
        self.protein_number = total_nodes - drug_number - disease_number

        self.unified_hetero_processor = (args.architecture_mode == 'MODE_A')

        # Homogeneous semantic network processors (SpectralSGE with frequency)
        self.drug_homo_sge = SpectralSGE(
            device, args.sge_layer, drug_number,
            args.sge_out_dim, args.sge_out_dim, args.sge_head, args.dropout,
            freq_enhance_ratio=args.freq_enhance_ratio,
            freq_low_pass_ratio=args.freq_low_pass_ratio,
            freq_min_components=args.freq_min_components,
            freq_max_components=args.freq_max_components
        )

        self.disease_homo_sge = SpectralSGE(
            device, args.sge_layer, disease_number,
            args.sge_out_dim, args.sge_out_dim, args.sge_head, args.dropout,
            freq_enhance_ratio=args.freq_enhance_ratio,
            freq_low_pass_ratio=args.freq_low_pass_ratio,
            freq_min_components=args.freq_min_components,
            freq_max_components=args.freq_max_components
        )

        # HSGE Architecture Configuration
        if args.architecture_mode == 'MODE_A':
            # Mode A: Single unified HSGE for both heterograph and metapaths
            self.unified_hsge = SpectralHSGE(
                meta_g=meta_g,
                hsge_out_dim=args.hsge_out_dim,
                hsge_head=args.hsge_head,
                hsge_layer=args.hsge_layer,
                dropout=args.dropout,
                freq_enhance_ratio=args.freq_enhance_ratio,
                freq_low_pass_ratio=args.freq_low_pass_ratio,
                freq_min_components=args.freq_min_components,
                freq_max_components=args.freq_max_components
            )
            self.hetero_hsge = None
            self.shared_path_hsge = None
        else:
            # Mode B: Separate HSGE for heterograph, shared HSGE for all metapaths
            self.hetero_hsge = SpectralHSGE(
                meta_g=meta_g,
                hsge_out_dim=args.hsge_out_dim,
                hsge_head=args.hsge_head,
                hsge_layer=args.hsge_layer,
                dropout=args.dropout,
                freq_enhance_ratio=args.freq_enhance_ratio,
                freq_low_pass_ratio=args.freq_low_pass_ratio,
                freq_min_components=args.freq_min_components,
                freq_max_components=args.freq_max_components
            )

            self.shared_path_hsge = SpectralHSGE(
                meta_g=meta_g,
                hsge_out_dim=args.path_hsge_out_dim,
                hsge_head=args.path_hsge_head,
                hsge_layer=args.path_hsge_layer,
                dropout=args.dropout,
                freq_enhance_ratio=args.freq_enhance_ratio,
                freq_low_pass_ratio=args.freq_low_pass_ratio,
                freq_min_components=args.freq_min_components,
                freq_max_components=args.freq_max_components
            )

            self.unified_hsge = None

        # Feature transformation layers for metapaths
        self.path_feature_transforms = None
        self._processors_initialized = False

        # Dimension adapters
        self._init_dimension_adapters()

    def _init_dimension_adapters(self):
        """Initialize dimension adapters to avoid dynamic parameter creation"""
        homo_dim = self.args.sge_out_dim

        if self.args.architecture_mode == 'MODE_A':
            path_dim = self.args.hsge_out_dim
        else:
            path_dim = self.args.path_hsge_out_dim

        self.drug_adapters = nn.ModuleDict()
        for path in self.path_names:
            if homo_dim != path_dim:
                self.drug_adapters[f'path_{path}'] = nn.Linear(path_dim, homo_dim)

        self.disease_adapters = nn.ModuleDict()
        for path in self.path_names:
            if homo_dim != path_dim:
                self.disease_adapters[f'path_{path}'] = nn.Linear(path_dim, homo_dim)

    def forward_homo_graphs(self, drdr_graph, didi_graph):
        """Process homogeneous semantic networks with SGE (includes frequency processing)"""
        dr_homo = self.drug_homo_sge(drdr_graph)
        di_homo = self.disease_homo_sge(didi_graph)
        return dr_homo, di_homo

    def forward_hetero_graph(self, heterograph, feature, node_types, edge_types):
        """Process heterogeneous regulatory network"""
        if self.args.architecture_mode == 'MODE_A':
            return self.unified_hsge(heterograph, feature, node_types, edge_types)
        else:
            return self.hetero_hsge(heterograph, feature, node_types, edge_types)

    def forward_path_graphs(self, path_data, drug_feature, disease_feature, protein_feature):
        """Process path-guided networks using HSGE only (no SGE option)"""
        if not self._processors_initialized:
            drug_dim = drug_feature.size(1)
            disease_dim = disease_feature.size(1)
            protein_dim = protein_feature.size(1)

            target_dim = max(drug_dim, disease_dim, protein_dim)

            self.path_feature_transforms = nn.ModuleDict({
                'drug': nn.Linear(drug_dim, target_dim).to(device),
                'disease': nn.Linear(disease_dim, target_dim).to(device),
                'protein': nn.Linear(protein_dim, target_dim).to(device)
            })

            self._processors_initialized = True

        path_embeddings = {}

        drug_feature_unified = self.path_feature_transforms['drug'](drug_feature)
        disease_feature_unified = self.path_feature_transforms['disease'](disease_feature)
        protein_feature_unified = self.path_feature_transforms['protein'](protein_feature)

        initial_features = torch.cat([drug_feature_unified, disease_feature_unified, protein_feature_unified], dim=0)

        for path in self.path_names:
            if path in path_data['path_adj_matrices']:
                adj_matrix = path_data['path_adj_matrices'][path]

                # Convert to heterograph for HSGE processing
                adj_sparse = sp.csr_matrix(adj_matrix)
                dgl_graph = dgl.from_scipy(adj_sparse).to(device)

                # Create fake node and edge types for homogeneous graph processing with SpectralHSGE
                num_nodes = dgl_graph.number_of_nodes()
                num_edges = dgl_graph.number_of_edges()
                node_types = torch.zeros(num_nodes, dtype=torch.long, device=device)
                edge_types = torch.zeros(num_edges, dtype=torch.long, device=device)

                # Use appropriate SpectralHSGE based on architecture mode
                if self.args.architecture_mode == 'MODE_A':
                    embedding = self.unified_hsge(dgl_graph, initial_features, node_types, edge_types)
                else:
                    embedding = self.shared_path_hsge(dgl_graph, initial_features, node_types, edge_types)

                path_embeddings[path] = embedding
            else:
                if self.args.architecture_mode == 'MODE_A':
                    embedding_dim = self.args.hsge_out_dim
                else:
                    embedding_dim = self.args.path_hsge_out_dim
                path_embeddings[path] = torch.zeros(self.total_nodes, embedding_dim).to(device)

        return path_embeddings


class MSGL(nn.Module):
    def __init__(self, meta_g, drug_number, disease_number, total_nodes, path_data=None):
        super(MSGL, self).__init__()
        self.drug_number = drug_number
        self.disease_number = disease_number
        self.total_nodes = total_nodes
        self.meta_g = meta_g
        self.path_data = path_data

        self.protein_number = total_nodes - drug_number - disease_number

        # Original feature linear transformation
        self.drug_linear = nn.Linear(300, args.hsge_out_dim)
        self.protein_linear = nn.Linear(320, args.hsge_out_dim)
        self.disease_linear = nn.Linear(64, args.hsge_out_dim)
        self.linear_initialized = False
        self.concat_drug = nn.Linear(args.hsge_out_dim * 3, args.hsge_out_dim)
        self.concat_disease = nn.Linear(args.hsge_out_dim * 3, args.hsge_out_dim)

        # Create unified node embedding for paths with correct dimension
        path_embedding_dim = args.hsge_out_dim if args.architecture_mode == 'MODE_A' else args.path_hsge_out_dim
        self.path_node_embedding = nn.Embedding(total_nodes, path_embedding_dim)
        nn.init.xavier_uniform_(self.path_node_embedding.weight)

        # Unified embedding module with HSGE architecture support
        self.unified_embedding = UnifiedEmbeddingModule(
            args=args,
            meta_g=meta_g,
            path_names=args.path_list,
            total_nodes=total_nodes,
            output_dim=args.hsge_out_dim if args.architecture_mode == 'MODE_A' else args.path_hsge_out_dim,
            drug_number=drug_number,
            disease_number=disease_number
        )

        # Attention fusion modules
        fusion_dim = args.sge_out_dim

        self.drug_attention_fusion = MultiSourceAttentionFusion(
            embedding_dim=fusion_dim,
            hidden_dim=args.attention_fusion_dim,
            num_heads=args.attention_fusion_heads,
            dropout=args.attention_dropout
        )

        self.disease_attention_fusion = MultiSourceAttentionFusion(
            embedding_dim=fusion_dim,
            hidden_dim=args.attention_fusion_dim,
            num_heads=args.attention_fusion_heads,
            dropout=args.attention_dropout
        )

        # Feature extractor
        self.feature_extractor = UnifiedFeatureExtractor()

        # MLP classifier
        mlp_input_dim = args.sge_out_dim

        self.mlp_classifier = MLPClassifier(
            input_dim=mlp_input_dim,
            hidden_dims=[1024, 1024, 256],
            num_classes=2,
            dropout=args.dropout,
            feature_combination='element_wise'
        )

        # Decoder (only type 1)
        decoder_dim = args.sge_out_dim

        encoder_layer = nn.TransformerEncoderLayer(d_model=decoder_dim, nhead=args.decoder_transformer_head)
        self.drug_trans = nn.TransformerEncoder(encoder_layer, num_layers=args.decoder_transformer_layer)
        self.disease_trans = nn.TransformerEncoder(encoder_layer, num_layers=args.decoder_transformer_layer)

        self.mlp = nn.Sequential(
            nn.Linear(decoder_dim, 512),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(128, 2))

        # Additional concat MLP
        total_concat_dim = fusion_dim * (1 + len(args.path_list))
        self.mlp_concat = nn.Sequential(
            nn.Linear(total_concat_dim, 1024),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(256, 2))

    def extract_node_features_by_range(self, full_embeddings, path_data):
        """
        Extract node features by range - segment by range, handling large-scale data
        """
        node_ranges = path_data['node_ranges']

        drug_start, drug_end = node_ranges['drug']
        disease_start, disease_end = node_ranges['disease']
        protein_start, protein_end = node_ranges['protein']

        drug_embeddings = full_embeddings[drug_start:drug_end]
        disease_embeddings = full_embeddings[disease_start:disease_end]
        protein_embeddings = full_embeddings[protein_start:protein_end]

        return drug_embeddings, disease_embeddings, protein_embeddings

    def forward(self, drdr_graph, didi_graph, drdipr_graph, drug_feature, disease_feature, protein_feature, sample,
                path_data=None, classifier_type='mlp'):
        # Dynamically adjust linear layer dimensions
        if not self.linear_initialized:
            drug_dim = drug_feature.shape[1]
            disease_dim = disease_feature.shape[1]
            protein_dim = protein_feature.shape[1]

            if drug_dim != 300:
                self.drug_linear = nn.Linear(drug_dim, args.hsge_out_dim).to(device)

            if disease_dim != 64:
                self.disease_linear = nn.Linear(disease_dim, args.hsge_out_dim).to(device)

            if protein_dim != 320:
                self.protein_linear = nn.Linear(protein_dim, args.hsge_out_dim).to(device)

            self.linear_initialized = True

        # 1. Homogeneous semantic network embedding (SGE with frequency processing)
        dr_homo, di_homo = self.unified_embedding.forward_homo_graphs(drdr_graph, didi_graph)

        # 2. Heterogeneous regulatory network aggregation embedding (HSGE with frequency processing)
        drug_feature_transformed = self.drug_linear(drug_feature)
        protein_feature_transformed = self.protein_linear(protein_feature)
        disease_feature_transformed = self.disease_linear(disease_feature)

        # Prepare features for HSGE
        feature_dict = {
            'drug': drug_feature_transformed,
            'disease': disease_feature_transformed,
            'protein': protein_feature_transformed
        }

        drdipr_graph.ndata['h'] = feature_dict
        g = dgl.to_homogeneous(drdipr_graph, ndata='h')

        feature = torch.cat((drug_feature_transformed, disease_feature_transformed, protein_feature_transformed),
                            dim=0)

        # Use unified embedding module for HSGE processing
        hsge_out = self.unified_embedding.forward_hetero_graph(g, feature, g.ndata['_TYPE'], g.edata['_TYPE'])

        dr_hgt = hsge_out[:self.drug_number, :]
        di_hgt = hsge_out[self.drug_number:self.drug_number + self.disease_number, :]

        # 3. Path-guided network embedding (SpectralHSGE only, with native frequency processing)
        path_embeddings_dict = {}
        drug_path_embeddings = {}
        disease_path_embeddings = {}
        if path_data:
            all_node_indices = torch.arange(self.total_nodes, device=device)
            all_path_features = self.path_node_embedding(all_node_indices)

            path_drug_feature = all_path_features[:self.drug_number]
            path_disease_feature = all_path_features[self.drug_number:self.drug_number + self.disease_number]
            path_protein_feature = all_path_features[self.drug_number + self.disease_number:]

            # Use SpectralHSGE for all metapaths (native frequency processing)
            path_embeddings_dict = self.unified_embedding.forward_path_graphs(
                path_data, path_drug_feature, path_disease_feature, path_protein_feature)

            for path, full_embedding in path_embeddings_dict.items():
                drug_emb, disease_emb, _ = self.extract_node_features_by_range(full_embedding, path_data)
                drug_path_embeddings[path] = drug_emb
                disease_path_embeddings[path] = disease_emb

        # 4. Attention fusion and concatenation fusion
        drug_embeddings_list = [dr_homo, dr_hgt]
        for path in args.path_list:
            if path in drug_path_embeddings:
                path_emb = drug_path_embeddings[path]
                if path_emb.size(-1) != dr_homo.size(-1):
                    adapter_key = f'path_{path}'
                    if adapter_key in self.unified_embedding.drug_adapters:
                        path_emb = self.unified_embedding.drug_adapters[adapter_key](path_emb)
                drug_embeddings_list.append(path_emb)

        disease_embeddings_list = [di_homo, di_hgt]
        for path in args.path_list:
            if path in disease_path_embeddings:
                path_emb = disease_path_embeddings[path]
                if path_emb.size(-1) != di_homo.size(-1):
                    adapter_key = f'path_{path}'
                    if adapter_key in self.unified_embedding.disease_adapters:
                        path_emb = self.unified_embedding.disease_adapters[adapter_key](path_emb)
                disease_embeddings_list.append(path_emb)

        # Attention fusion
        if len(drug_embeddings_list) > 1:
            dr_fused = self.drug_attention_fusion(drug_embeddings_list)
            di_fused = self.disease_attention_fusion(disease_embeddings_list)
        else:
            dr_fused = drug_embeddings_list[0]
            di_fused = disease_embeddings_list[0]

        # Concatenation fusion
        dr_concat = torch.cat(drug_embeddings_list, dim=-1)
        di_concat = torch.cat(disease_embeddings_list, dim=-1)
        concat = False
        if concat == True:
            dr_concat = self.concat_drug(dr_concat)
            di_concat = self.concat_disease(di_concat)

        # 5. Feature extraction and classification
        feature_dict = self.feature_extractor.extract_pair_features(
            dr_fused, di_fused, sample, feature_type='all'
        )

        # Use decoder type 1 only
        decoder_dim = args.sge_out_dim
        dr_final = dr_fused.view(self.drug_number, decoder_dim)
        di_final = di_fused.view(self.disease_number, decoder_dim)

        drdi_embedding = torch.mul(dr_final[sample[:, 0]], di_final[sample[:, 1]])

        output = self.mlp(drdi_embedding)

        # Return results including all required embeddings for contrastive learning
        result_dict = {
            'prediction': output,
            'homo_embeddings': (dr_homo, di_homo),
            'hgt_embeddings': (dr_hgt, di_hgt),
            'path_embeddings': (drug_path_embeddings, disease_path_embeddings),
            'fused_embeddings': (dr_fused, di_fused),
            'concat_embeddings': (dr_concat, di_concat),
            'feature_dict': None
        }

        return result_dict

    def load(self, checkpoint_dir):
        self.load_state_dict(torch.load(checkpoint_dir))