# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn.functional as F

EPSILON = float(np.finfo(float).eps)


def safe_log(x):
    return torch.log(x + EPSILON)


def get_pos_neg_indice(drug_simlilarity_matrix, disease_simlilarity_matrix):
    drug_num = drug_simlilarity_matrix.shape[0]
    disease_num = disease_simlilarity_matrix.shape[0]

    drug_pos_indice = drug_simlilarity_matrix - np.eye(drug_num)
    disease_pos_indice = disease_simlilarity_matrix - np.eye(disease_num)

    drug_pos_indice = torch.from_numpy(drug_pos_indice).long().cuda()
    disease_pos_indice = torch.from_numpy(disease_pos_indice).long().cuda()

    drug_neg_indice = torch.from_numpy(drug_simlilarity_matrix).long().cuda()
    drug_neg_indice = (drug_neg_indice == 0).long()

    disease_neg_indice = torch.from_numpy(disease_simlilarity_matrix).long().cuda()
    disease_neg_indice = (disease_neg_indice == 0).long()

    return drug_pos_indice, drug_neg_indice, disease_pos_indice, disease_neg_indice


def semantic_regulatory_alignment_loss(drug_simlilarity_matrix, disease_simlilarity_matrix, homo_drug_feature,
                                       homo_disease_feature, hetero_drug_feature, hetero_disease_feature, temperature):
    """Semantic-regulatory alignment contrastive learning"""
    _, drug_neg_indice, _, disease_neg_indice = get_pos_neg_indice(drug_simlilarity_matrix, disease_simlilarity_matrix)

    homo_drug_feature = F.normalize(homo_drug_feature, p=2, dim=1)
    homo_disease_feature = F.normalize(homo_disease_feature, p=2, dim=1)
    hetero_drug_feature = F.normalize(hetero_drug_feature, p=2, dim=1)
    hetero_disease_feature = F.normalize(hetero_disease_feature, p=2, dim=1)

    drug_pos_score = torch.multiply(homo_drug_feature, hetero_drug_feature).sum(dim=1)
    drug_neg_score = torch.matmul(homo_drug_feature, hetero_drug_feature.t())

    disease_pos_score = torch.multiply(homo_disease_feature, hetero_disease_feature).sum(dim=1)
    disease_neg_score = torch.matmul(homo_disease_feature, hetero_disease_feature.t())

    drug_neg_score = drug_neg_indice * drug_neg_score
    disease_neg_score = disease_neg_indice * disease_neg_score

    drug_pos_score = torch.exp(drug_pos_score / temperature)
    drug_neg_score = torch.exp(drug_neg_score / temperature).sum(dim=1)

    disease_pos_score = torch.exp(disease_pos_score / temperature)
    disease_neg_score = torch.exp(disease_neg_score / temperature).sum(dim=1)

    drug_ssl_loss = -torch.log(drug_pos_score / drug_neg_score).sum()
    disease_ssl_loss = -torch.log(disease_pos_score / disease_neg_score).sum()

    return drug_ssl_loss + disease_ssl_loss


def compute_total_contrastive_loss(drug_simlilarity_matrix, disease_simlilarity_matrix,
                                   homo_drug_feature, homo_disease_feature,
                                   hetero_drug_feature, hetero_disease_feature,
                                   semantic_regulatory_temperature=0.05,
                                   semantic_regulatory_weight=0.00001):
    """
    Compute total contrastive learning loss for MSGL framework

    Args:
        drug_simlilarity_matrix: Drug similarity matrix
        disease_simlilarity_matrix: Disease similarity matrix
        homo_drug_feature: Homogeneous semantic network drug embeddings
        homo_disease_feature: Homogeneous semantic network disease embeddings
        hetero_drug_feature: Heterogeneous regulatory network drug embeddings
        hetero_disease_feature: Heterogeneous regulatory network disease embeddings
        semantic_regulatory_temperature: Temperature for semantic-regulatory alignment
        semantic_regulatory_weight: Weight for semantic-regulatory alignment loss

    Returns:
        loss_dict: Dictionary containing various losses
    """
    loss_dict = {}

    semantic_regulatory_loss = semantic_regulatory_alignment_loss(
        drug_simlilarity_matrix, disease_simlilarity_matrix,
        homo_drug_feature, homo_disease_feature,
        hetero_drug_feature, hetero_disease_feature,
        semantic_regulatory_temperature
    )
    loss_dict['semantic_regulatory_loss'] = semantic_regulatory_loss

    total_contrastive_loss = semantic_regulatory_weight * semantic_regulatory_loss

    loss_dict['total_contrastive_loss'] = total_contrastive_loss

    return loss_dict