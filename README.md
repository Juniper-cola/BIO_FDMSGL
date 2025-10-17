# FD-MSGL: Drug Repositioning via Frequency-Domain Multi-Source Synergistic Graph Learning

The FD-MSGL framework addresses drug repositioning through multi-scale biological network modeling in the frequency domain. The framework comprises three complementary biological networks that capture different aspects of drug-disease relationships. Homogeneous semantic networks encode drug-drug and disease-disease similarity patterns based on chemical structure and phenotypic relationships. Heterogeneous regulatory networks model direct drug-protein-disease molecular interactions, capturing target selectivity profiles and pathological mechanisms. Pathway-guided networks capture indirect regulatory mechanisms through Drug-Target-Disease (DTD), Drug-Target-Protein-Target (DTPT), and Drug-Disease-Target-Protein (DDTP) cascade patterns. Through spectral decomposition that decomposes these networks into frequency components, the model simultaneously captures local molecular recognition precision and global systemic coordination patterns. This multi-scale representation enables the identification of repositioning candidates that act through both direct target binding and complex multi-step regulatory pathways, providing systematic biological evidence for drug discovery decisions.
<img width="1582" height="1329" alt="image" src="https://github.com/user-attachments/assets/12d23950-adc7-4180-9e0b-ab5c012bde28" />
## Data Requirements

The FD-MSGL framework is designed with a general multi-relational graph learning architecture that extends beyond biological applications. While originally developed for drug repositioning, the framework can accommodate any domain-specific data that conforms to the required structural format. The core requirement is a multi-entity heterogeneous network with three types of nodes and their pairwise associations, along with node-level feature representations and similarity measures.
For biological applications, the framework processes data representing drugs, diseases, and proteins along with their associations. Drug information includes molecular fingerprints and chemical similarity profiles. Disease information encompasses phenotypic similarity measures and characteristic feature representations. Protein data provides functional embeddings and interaction profiles. The association data captures known drug-disease relationships, drug-protein interactions, and protein-disease connections in matrix format.
For applications in other domains, the three-entity structure can be adapted accordingly. The framework expects entity features as numerical vectors, pairwise similarity matrices for each entity type, and association matrices defining relationships between entity pairs. Input data should be organized as CSV files containing feature matrices and association networks. The preprocessing pipeline constructs similarity graphs through k-nearest neighbor algorithms and generates heterogeneous networks from the association matrices. Pathway data is derived through metapath-based random walk strategies that extract structurally meaningful subgraph patterns from the integrated network.

## Requirements

The implementation has been tested with the following environment configuration:

- Python 3.7.13
- PyTorch 1.13.1 with CUDA 11.7
- DGL 0.9.1 (CUDA 11.7)
- PyTorch Geometric 2.3.1
- NumPy 1.21.6
- Pandas 1.3.5
- SciPy 1.7.3
- scikit-learn 0.24.2
- NetworkX 2.6.3
- tqdm 4.67.1

The framework requires GPU support for efficient training and inference. CUDA 11.7 or compatible versions are necessary for leveraging the spectral graph processing capabilities.

## Usage

The framework can be executed through the main training script with configurable parameters for different experimental settings. The basic training command requires specification of the dataset and GPU device:

```bash
python main.py --dataset B-dataset --gpu 0
```

For customized experimental configurations, additional parameters control the architecture mode, training duration, and cross-validation strategy:

```bash
python main.py --dataset B-dataset --gpu 0 --total_epochs 1000 --K_fold 10 --architecture_mode MODE_A ...
```

## Citation

If you use this code in your research, please cite our paper:

```
FD-MSGL: Drug Repositioning via Frequency-Domain Multi-Source Synergistic Graph Learning
```

Citation details will be updated upon publication.
