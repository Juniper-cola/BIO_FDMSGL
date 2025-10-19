# -*- coding: utf-8 -*-
import os
import sys
import argparse

argparser = argparse.ArgumentParser(sys.argv[0])

argparser.add_argument("--dataset",
                       type=str,
                       default='B-dataset',
                       help="dataset for training")

argparser.add_argument('--gpu', type=int, default=0,
                       help='gpu device')

argparser.add_argument('--seed', type=int, default=2025, help='random seed')

argparser.add_argument('--K_fold', type=int, default=10, help='k-fold cross validation')

argparser.add_argument('--negative_rate', type=float, default=1.0, help='negative_rate')

argparser.add_argument('--KNN_neighbor', type=int, default=25, help='neighbor_num')

argparser.add_argument('--total_epochs', type=int, default=1000,
                       help='epoch number.')

argparser.add_argument('--dataset_percent', type=float, default=1)

argparser.add_argument('--sge_layer', default=2, type=int, help='SGE layer for homogeneous semantic networks')

argparser.add_argument('--sge_head', default=4, type=int, help='SGE head for homogeneous semantic networks')

argparser.add_argument('--sge_out_dim', default=256, type=int,
                       help='SGE output dimension for homogeneous semantic networks')

argparser.add_argument('--path_hsge_layer', default=2, type=int,
                       help='HSGE layer for path-guided network embedding')

argparser.add_argument('--path_hsge_head', default=4, type=int,
                       help='HSGE head for path-guided network embedding')

argparser.add_argument('--path_hsge_out_dim', default=256, type=int,
                       help='HSGE output dimension for path-guided network embedding')

argparser.add_argument('--path_walk_length', type=int, default=10,
                       help='regulatory path random walk length for subgraph generation')

argparser.add_argument('--path_num_walks', type=int, default=30,
                       help='number of random walks per node for subgraph generation')

argparser.add_argument('--path_list', nargs='+', default=["DTD", "DTPT", "DDTP"],
                       help='list of regulatory paths for embedding generation')

argparser.add_argument('--hsge_layer', default=2, type=int, help='heterogeneous SGE layer')

argparser.add_argument('--hsge_head', default=4, type=int, help='heterogeneous SGE head')

argparser.add_argument('--hsge_out_dim', default=256, type=int, help='heterogeneous SGE output dimension')

argparser.add_argument('--hsge_in_dim', default=256, type=int, help='heterogeneous SGE input dimension')

argparser.add_argument('--freq_enhance_ratio', type=float, default=0.4,
                       help='ratio of frequency enhancement (alpha parameter for spatial-frequency mixing)')

argparser.add_argument('--freq_low_pass_ratio', type=float, default=0.3,
                       help='ratio of eigenvalues to keep for low-pass filtering')

argparser.add_argument('--freq_min_components', type=int, default=64,
                       help='minimum number of frequency components to keep')

argparser.add_argument('--freq_max_components', type=int, default=128,
                       help='maximum number of frequency components to keep')

argparser.add_argument('--architecture_mode', type=str, default='MODE_A',
                       choices=['MODE_A', 'MODE_B'],
                       help='MODE_A: unified HSGE for both heterograph and metapaths; MODE_B: separate HSGE for heterograph, shared HSGE for metapaths')

argparser.add_argument('--dropout', default=0.4, type=float, help='dropout')

argparser.add_argument('--weight_decay', type=float, default=1e-6, help='weight_decay')

argparser.add_argument('--lr', type=float, default=5e-5, help='learning rate')

argparser.add_argument('--decoder_type', default=1, type=int, help='decoder type (fixed to 1)')

argparser.add_argument('--decoder_transformer_layer', default=2, type=int, help='transformer layer for decoder')

argparser.add_argument('--decoder_transformer_head', default=4, type=int, help='transformer head for decoder')

argparser.add_argument('--semantic_regulatory_temperature', type=float, default=0.05,
                       help='temperature for semantic-regulatory alignment contrastive learning')

argparser.add_argument('--semantic_regulatory_weight', type=float, default=0.00001,
                       help='weight for semantic-regulatory alignment loss')

argparser.add_argument('--attention_fusion_dim', default=256, type=int,
                       help='attention fusion hidden dimension')

argparser.add_argument('--attention_fusion_heads', default=4, type=int,
                       help='number of attention heads for fusion')

argparser.add_argument('--attention_dropout', default=0.1, type=float,
                       help='dropout rate for attention fusion')

argparser.add_argument('--print_interval', type=int, default=50,
                       help='interval for printing training progress')

argparser.add_argument('--eval_interval', type=int, default=1,
                       help='interval for evaluation during training')

argparser.add_argument('--feature_combination', type=str, default='concat',
                       choices=['concat', 'element_wise', 'hadamard', 'advanced'],
                       help='feature combination method for classification')

argparser.add_argument('--homo_processor', type=str, default='sge',
                       choices=['sge'],
                       help='homogeneous semantic network processor: sge only')

argparser.add_argument('--path_processor', type=str, default='hsge',
                       choices=['hsge'],
                       help='path-guided network processor: hsge only (metapaths require heterogeneous processing)')

argparser.add_argument('--path_embedding_dim', type=int, default=256,
                       help='embedding dimension for path-guided network node features')

args = argparser.parse_args()

if not hasattr(args, 'hsge_in_dim'):
    args.hsge_in_dim = args.hsge_out_dim
args.path_processor = 'hsge'

args.unified_hetero_processor = (args.architecture_mode == 'MODE_A')

if args.architecture_mode == 'MODE_A':
    args.path_hsge_out_dim = args.hsge_out_dim
    args.path_hsge_layer = args.hsge_layer
    args.path_hsge_head = args.hsge_head
    print("Architecture Mode: A (Unified HSGE)")
    print("  - Single HSGE processes both heterograph and all metapaths")
    print("  - Parameter sharing between heterograph and metapath processing")
    print("  - HSGE dimensions: {}".format(args.hsge_out_dim))
elif args.architecture_mode == 'MODE_B':
    print("Architecture Mode: B (Separated HSGE)")
    print("  - Heterograph HSGE dimensions: {}".format(args.hsge_out_dim))
    print("  - Shared metapath HSGE dimensions: {}".format(args.path_hsge_out_dim))
    print("  - All metapaths share the same HSGE parameters")

if args.freq_enhance_ratio < 0 or args.freq_enhance_ratio > 1:
    raise ValueError("freq_enhance_ratio must be between 0 and 1")
if args.freq_low_pass_ratio <= 0 or args.freq_low_pass_ratio > 1:
    raise ValueError("freq_low_pass_ratio must be between 0 and 1")
if args.freq_min_components <= 0:
    raise ValueError("freq_min_components must be positive")
if args.freq_max_components < args.freq_min_components:
    raise ValueError("freq_max_components must be >= freq_min_components")

print("Frequency Domain Processing (Native in SpectralSGE/SpectralHSGE):")
print("  - Enhancement ratio: {}".format(args.freq_enhance_ratio))
print("  - Low-pass filter ratio: {}".format(args.freq_low_pass_ratio))
print("  - Component range: {}-{}".format(args.freq_min_components, args.freq_max_components))

print("Final Architecture Configuration:")
print("  - Homogeneous Semantic Networks: SpectralSGE (native frequency domain)")
print("  - Heterogeneous Regulatory Network: SpectralHSGE (native frequency domain)")
print("  - Path-guided Networks: SpectralHSGE (native frequency domain)")
print("  - Metapath Processing: {} SpectralHSGE mode".format(
    "Unified" if args.architecture_mode == 'MODE_A' else "Shared Separate"))
print("  - Contrastive Learning: Semantic-Regulatory Alignment")