# -*- coding: utf-8 -*-
import os
import torch
import itertools
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from model import MSGL
import torch.optim as optim
from parse_args import args
from metric import get_metric
from datetime import datetime
import torch.nn.functional as fn
from torch_geometric.seed import seed_everything
from data_preprocessing import process_data, dgl_heterograph
from contrastive_learning import compute_total_contrastive_loss
from metapath_generator import generate_fold_path_data
import logging
import warnings
import time

warnings.filterwarnings('ignore')
torch.autograd.set_detect_anomaly(True)


def setup_logging(args):
    """Setup logging system"""
    if not os.path.exists('logs'):
        os.makedirs('logs')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    processor_config = f"SpectralSGE_SpectralHSGE_{args.path_processor.upper()}"

    arch_info = "UNIFIED" if args.architecture_mode == 'MODE_A' else "SEPARATE"

    log_filename = 'logs/train_DDA_{}_{}_SPECTRAL_{}_{}.log'.format(
        args.dataset, processor_config, arch_info, timestamp
    )

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("MSGL (Multi-Source Synergistic Graph Learning) Training Session Started")
    logger.info("=" * 80)

    logger.info("COMPLETE MODEL CONFIGURATION:")
    logger.info("-" * 50)
    args_dict = vars(args)
    for key in sorted(args_dict.keys()):
        logger.info("{}:{:30}".format(key, str(args_dict[key])))
    logger.info("-" * 50)

    logger.info("KEY ARCHITECTURE CHOICES:")
    logger.info("  Homogeneous Semantic Network Processor: SpectralSGE")
    logger.info("  Path-guided Network Processor: SpectralHSGE")
    logger.info(
        "  Heterogeneous Architecture Mode: {}".format("Unified" if args.architecture_mode == 'MODE_A' else "Separated"))
    logger.info("  Classifier Type: MLP")
    logger.info("  Decoder Type: {}".format(args.decoder_type))

    logger.info("FREQUENCY DOMAIN PROCESSING (NATIVE):")
    logger.info("  Status: Always Enabled (Native in SpectralSGE/SpectralHSGE)")
    logger.info("  Enhancement Ratio: {}".format(args.freq_enhance_ratio))
    logger.info("  Low-pass Filter Ratio: {}".format(args.freq_low_pass_ratio))
    logger.info("  Component Range: {}-{}".format(args.freq_min_components, args.freq_max_components))

    logger.info("HETEROGENEOUS PROCESSING ARCHITECTURE:")
    if args.architecture_mode == 'MODE_A':
        logger.info("  Mode: Unified Processing")
        logger.info("  Description: Single SpectralHSGE for both heterograph and metapaths")
        logger.info("  Metapath Processor: Unified SpectralHSGE")
    else:
        logger.info("  Mode: Separated Processing")
        logger.info("  Description: Separate processors for heterograph and metapaths")
        logger.info("  Heterograph Processor: Dedicated SpectralHSGE")
        logger.info("  Metapath Processor: Shared SpectralHSGE for all paths")

    logger.info("=" * 80)

    return logger, log_filename


def log_fold_results(logger, fold_idx, metrics_dict, best_epoch, fold_time):
    """Log detailed results for each fold"""
    logger.info("-" * 80)
    logger.info("FOLD {} RESULTS:".format(fold_idx + 1))
    logger.info("-" * 80)
    logger.info("Training Time: {:.2f} seconds".format(fold_time))
    logger.info("Best Epoch: {}".format(best_epoch))
    logger.info("  AUC:       {:.5f}".format(metrics_dict['AUC']))
    logger.info("  AUPR:      {:.5f}".format(metrics_dict['AUPR']))
    logger.info("  Accuracy:  {:.5f}".format(metrics_dict['Accuracy']))
    logger.info("  Precision: {:.5f}".format(metrics_dict['Precision']))
    logger.info("  Recall:    {:.5f}".format(metrics_dict['Recall']))
    logger.info("  F1-score:  {:.5f}".format(metrics_dict['F1']))
    logger.info("  MCC:       {:.5f}".format(metrics_dict['MCC']))
    logger.info("-" * 80)


def log_final_results(logger, metrics_collection):
    """Log final statistics"""
    logger.info("=" * 100)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("=" * 100)

    logger.info("Metric      | Mean +/- Std")
    logger.info("------------|------------------")

    final_stats = {}
    for metric_name in ['AUC', 'AUPR', 'Accuracy', 'Precision', 'Recall', 'F1', 'MCC']:
        values = metrics_collection[metric_name]
        mean_val = np.mean(values)
        std_val = np.std(values)
        final_stats[metric_name] = {'mean': mean_val, 'std': std_val, 'values': values}
        logger.info("{:11} | {:.5f} +/- {:.5f}".format(metric_name, mean_val, std_val))

    logger.info("=" * 100)
    logger.info("DETAILED FOLD-BY-FOLD RESULTS:")
    logger.info("=" * 100)

    num_folds = len(metrics_collection['AUC'])
    for fold_idx in range(num_folds):
        auc_val = metrics_collection['AUC'][fold_idx]
        aupr_val = metrics_collection['AUPR'][fold_idx]
        mcc_val = metrics_collection['MCC'][fold_idx]
        logger.info("Fold {:2d}: AUC={:.5f}, AUPR={:.5f}, MCC={:.5f}".format(
            fold_idx + 1, auc_val, aupr_val, mcc_val))

    return final_stats


def print_console_summary(metrics_collection):
    """Print concise summary to console"""
    print("\n" + "=" * 100)
    print("FINAL RESULTS SUMMARY")
    print("=" * 100)

    print("Metric      | Mean +/- Std")
    print("------------|------------------")

    for metric_name in ['AUC', 'AUPR', 'Accuracy', 'Precision', 'Recall', 'F1', 'MCC']:
        values = metrics_collection[metric_name]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print("{:11} | {:.5f} +/- {:.5f}".format(metric_name, mean_val, std_val))

    print("=" * 100)


def train_mlp(model, drdr_graph, didi_graph, drdipr_graph, drug_feature, disease_feature, protein_feature,
              X_train, X_test, Y_train, Y_test, drdr_matrix, didi_matrix, path_data, logger, fold_idx):
    cross_entropy = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)

    best_mixed_metric = -float("inf")
    best_mixed_result = None

    l_reports = []

    best_auc = 0.0
    best_aupr = 0.0
    epoch_start_time = time.time()

    arch_status = "UNIFIED" if args.architecture_mode == 'MODE_A' else "SEPARATE"

    pbar = tqdm(range(args.total_epochs),
                desc="Fold {} MSGL Training (SpectralSGE+SpectralHSGE, SPECTRAL, {})".format(
                    fold_idx + 1, arch_status
                ),
                ncols=140,
                leave=True)

    for epoch in pbar:
        model.train()

        result_dict = model(drdr_graph, didi_graph, drdipr_graph,
                            drug_feature, disease_feature, protein_feature,
                            X_train, path_data, classifier_type='mlp')

        train_score = result_dict['prediction']
        homo_drug, homo_disease = result_dict['homo_embeddings']
        hetero_drug, hetero_disease = result_dict['hgt_embeddings']

        train_loss = cross_entropy(train_score, torch.flatten(Y_train))

        contrastive_losses = compute_total_contrastive_loss(
            drdr_matrix, didi_matrix,
            homo_drug, homo_disease,
            hetero_drug, hetero_disease,
            semantic_regulatory_temperature=args.semantic_regulatory_temperature,
            semantic_regulatory_weight=args.semantic_regulatory_weight
        )

        total_loss = train_loss + contrastive_losses['total_contrastive_loss']

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if use_cuda:
            torch.cuda.empty_cache()

        if epoch % args.eval_interval == 0 or epoch == args.total_epochs - 1:
            with torch.no_grad():
                model.eval()
                test_result_dict = model(drdr_graph, didi_graph, drdipr_graph,
                                         drug_feature, disease_feature, protein_feature,
                                         X_test, path_data, classifier_type='mlp')
                test_score = test_result_dict['prediction']

            test_prob = fn.softmax(test_score, dim=-1)
            test_pred = torch.argmax(test_score, dim=-1)

            test_prob = test_prob[:, 1]
            test_prob = test_prob.cpu().numpy()
            test_pred = test_pred.cpu().numpy()

            AUC, AUPR, accuracy, precision, recall, f1, mcc = get_metric(Y_test, test_pred, test_prob)

            if AUC > best_auc:
                best_auc = AUC
            if AUPR > best_aupr:
                best_aupr = AUPR

            current_result = [epoch + 1, round(AUC, 5), round(AUPR, 5), round(accuracy, 5),
                              round(precision, 5), round(recall, 5), round(f1, 5), round(mcc, 5)]
            l_reports.append("\t".join(map(str, current_result)) + "\n")

            pbar.set_postfix({
                'AUC': '{:.4f}'.format(AUC),
                'AUPR': '{:.4f}'.format(AUPR),
                'Best_AUC': '{:.4f}'.format(best_auc),
                'Best_AUPR': '{:.4f}'.format(best_aupr)
            })

            mix_factor = AUC + AUPR + mcc
            if mix_factor > best_mixed_metric:
                best_mixed_metric = mix_factor
                best_mixed_result = current_result

    pbar.close()

    return best_mixed_result


def calculate_regulatory_path_total_nodes(d_data):
    """Calculate node allocation for regulatory path generation according to Q2 consecutive ID assignment"""
    drug_count = d_data['drug_number']
    disease_count = d_data['disease_number']
    protein_count = d_data['protein_number']

    total_nodes = drug_count + disease_count + protein_count

    print("Regulatory path node allocation:")
    print("  Drug nodes: 0 ~ {}".format(drug_count - 1))
    print("  Disease nodes: {} ~ {}".format(drug_count, drug_count + disease_count - 1))
    print("  Protein nodes: {} ~ {}".format(drug_count + disease_count, drug_count + disease_count + protein_count - 1))
    print("  Total nodes: {}".format(total_nodes))

    return total_nodes


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cuda = torch.cuda.is_available()
    torch.cuda.set_device(args.gpu)

    seed_everything(args.seed)

    logger, log_filename = setup_logging(args)

    print("=== Processing Base Data (MSGL Framework) ===")
    d_data, drdr_graph, didi_graph = process_data()

    total_nodes = calculate_regulatory_path_total_nodes(d_data)

    args.drug_number = d_data['drug_number']
    args.disease_number = d_data['disease_number']
    args.protein_number = d_data['protein_number']
    args.total_nodes = total_nodes

    drdr_graph = drdr_graph.to(device)
    didi_graph = didi_graph.to(device)

    logger.info("Data loaded - Drugs: {}, Diseases: {}, Proteins: {}, Total: {}".format(
        d_data['drug_number'], d_data['disease_number'], d_data['protein_number'], total_nodes))

    metrics_collection = {
        'AUC': [], 'AUPR': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1': [], 'MCC': []
    }

    l_results = []

    logger.info("Starting {}-fold cross validation...".format(args.K_fold))
    logger.info("MSGL Architecture Configuration:")
    logger.info("  - Homogeneous Semantic Networks: SpectralSGE")
    logger.info("  - Path-guided Networks: SpectralHSGE")
    logger.info("  - Heterogeneous Regulatory Network: SpectralHSGE")
    logger.info(
        "  - Heterogeneous Processing Mode: {}".format("Unified" if args.architecture_mode == 'MODE_A' else "Separated"))
    logger.info("  - Paths: {}".format(args.path_list))
    logger.info("  - Classifier: MLP")
    logger.info("  - Frequency Processing: Native (always enabled)")
    logger.info("    - Enhancement Ratio: {}".format(args.freq_enhance_ratio))
    logger.info("    - Low-pass Ratio: {}".format(args.freq_low_pass_ratio))
    logger.info("  - Contrastive Learning: Semantic-Regulatory Alignment")

    for fold_i in range(args.K_fold):
        fold_start_time = time.time()

        logger.info("Starting Fold {}/{}".format(fold_i + 1, args.K_fold))
        print("\n{}".format("=" * 70))
        print("FOLD {}/{} - MSGL Framework".format(fold_i + 1, args.K_fold))
        print("  Semantic: SpectralSGE | Path-guided: SpectralHSGE | Regulatory: SpectralHSGE")
        print("  Architecture: {} | Frequency: Native".format(
            "Unified" if args.architecture_mode == 'MODE_A' else "Separated"))
        print("  Contrastive: Semantic-Regulatory Alignment")
        print("{}".format("=" * 70))

        X_train_fold = d_data['X_train'][fold_i]
        Y_train_fold = d_data['Y_train'][fold_i].flatten()

        positive_mask = Y_train_fold == 1
        negative_mask = Y_train_fold == 0

        positive_indices = np.where(positive_mask)[0]
        negative_indices = np.where(negative_mask)[0]

        num_positive = int(args.dataset_percent * len(positive_indices))
        num_negative = int(args.dataset_percent * len(negative_indices))

        selected_positive = positive_indices[:num_positive]
        selected_negative = negative_indices[:num_negative]

        np_X_train_fold_i_positive = X_train_fold[selected_positive]
        np_Y_train_fold_i_positive = Y_train_fold[selected_positive]

        np_X_train_fold_i_negative = X_train_fold[selected_negative]
        np_Y_train_fold_i_negative = Y_train_fold[selected_negative]

        np_X_train_fold_i = np.concatenate([np_X_train_fold_i_positive, np_X_train_fold_i_negative], axis=0)
        np_Y_train_fold_i = np.concatenate([np_Y_train_fold_i_positive, np_Y_train_fold_i_negative], axis=0)

        X_train = torch.LongTensor(np_X_train_fold_i).to(device)
        Y_train = torch.LongTensor(np_Y_train_fold_i).to(device)
        X_test = torch.LongTensor(d_data['X_test'][fold_i]).to(device)
        Y_test = d_data['Y_test'][fold_i].flatten()

        print("=== Generating Fold-Specific Regulatory Path Data ===")
        path_data_generation_start = time.time()

        path_data = generate_fold_path_data(
            train_drdi_edges=np_X_train_fold_i_positive,
            d_data=d_data,
            args=args,
            device=device,
            fold_idx=fold_i
        )

        path_data_generation_time = time.time() - path_data_generation_start
        print("Regulatory path data generation time: {:.2f}s".format(path_data_generation_time))
        logger.info(
            "Fold {} regulatory path data generation time: {:.2f}s".format(fold_i + 1, path_data_generation_time))

        heterograph = dgl_heterograph(d_data, np_X_train_fold_i_positive)
        heterograph = heterograph.to(device)
        meta_g = heterograph.metagraph()

        model = MSGL(meta_g, d_data['drug_number'], d_data['disease_number'], total_nodes, path_data)
        model = model.to(device)

        drdr_graph = drdr_graph.to(device)
        didi_graph = didi_graph.to(device)
        heterograph = heterograph.to(device)

        drug_feature = torch.FloatTensor(d_data['drugfeature']).to(device)
        disease_feature = torch.FloatTensor(d_data['diseasefeature']).to(device)
        protein_feature = torch.FloatTensor(d_data['proteinfeature']).to(device)

        drdr_matrix_bool = d_data["drdr_matrix"]
        didi_matrix_bool = d_data["didi_matrix"]

        print("Starting training for Fold {} with MSGL architecture...".format(fold_i + 1))

        logger.info("Fold {} Model Architecture Details:".format(fold_i + 1))
        logger.info("  - Frequency Processing: Native (SpectralSGE/SpectralHSGE)")
        logger.info(
            "  - Heterogeneous Processing: {}".format("Unified" if args.architecture_mode == 'MODE_A' else "Separated"))
        logger.info("  - Path Processor: SpectralHSGE")

        fold_best_result = train_mlp(model, drdr_graph, didi_graph, heterograph,
                                      drug_feature, disease_feature, protein_feature,
                                      X_train, X_test, Y_train, Y_test,
                                      drdr_matrix_bool, didi_matrix_bool,
                                      path_data, logger, fold_i)

        if fold_best_result is not None:
            fold_end_time = time.time()
            fold_time = fold_end_time - fold_start_time

            best_epoch = fold_best_result[0]
            fold_results = fold_best_result[1:]
            l_results.append(fold_results)

            fold_metrics = {
                'AUC': fold_results[0], 'AUPR': fold_results[1], 'Accuracy': fold_results[2],
                'Precision': fold_results[3], 'Recall': fold_results[4], 'F1': fold_results[5],
                'MCC': fold_results[6]
            }

            for i, metric_name in enumerate(['AUC', 'AUPR', 'Accuracy', 'Precision', 'Recall', 'F1', 'MCC']):
                metrics_collection[metric_name].append(fold_results[i])

            log_fold_results(logger, fold_i, fold_metrics, best_epoch, fold_time)

            print("\nFold {} Results: {}".format(fold_i + 1, fold_best_result))

        if use_cuda:
            torch.cuda.empty_cache()

    final_stats = log_final_results(logger, metrics_collection)
    print_console_summary(metrics_collection)

    if l_results:
        np_results = np.array(l_results)
        mean_result = np.round(np.mean(np_results, axis=0), 4)
        var_result = np.round(np.var(np_results, axis=0), 4)
        std_result = np.round(np.std(np_results, axis=0), 4)

        print("\nFinal Statistics:")
        print("Mean: {}".format(mean_result))
        print("Variance: {}".format(var_result))
        print("Std: {}".format(std_result))

    logger.info("Results saved to: {}".format(log_filename))
    logger.info("MSGL training completed successfully!")

    print("\n{}".format("=" * 70))
    print("MSGL TRAINING CONFIGURATION SUMMARY")
    print("{}".format("=" * 70))
    print("Dataset: {}".format(args.dataset))
    print("Architecture: SpectralSGE + SpectralHSGE + SpectralHSGE")
    print("Heterogeneous Mode: {}".format("Unified" if args.architecture_mode == 'MODE_A' else "Separated"))
    print("Frequency Processing: Native (always enabled)")
    print("  - Enhancement Ratio: {}".format(args.freq_enhance_ratio))
    print("  - Low-pass Filter Ratio: {}".format(args.freq_low_pass_ratio))
    print("Paths: {}".format(args.path_list))
    print("Classifier: MLP")
    print("Decoder Type: {}".format(args.decoder_type))
    print("Epochs: {}".format(args.total_epochs))
    print("K-Fold: {}".format(args.K_fold))
    print("GPU: {}".format(args.gpu))
    print("Seed: {}".format(args.seed))
    print("Total Nodes: {}".format(total_nodes))
    print("Contrastive Learning: Semantic-Regulatory Alignment")
    print("{}".format("=" * 70))