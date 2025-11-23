# stkan

Variational autoencoder with Kolmogorov-Arnold Network for spatial domain detection.

Experiments were executed on NVIDIA A40 of 46068MiB memory in linux with torch==2.1.0+cu121, torch_geometric==2.3.1, torch-sparse==0.6.18+pt21cu121, and torchvision==0.16.0+cu121.

## overview

The stkan is an integrated framework for spatial domain detection in spatial transcriptomics that synergizes the Kolmogorov-Arnold Network (KAN) with a variational autoencoder and contrastive learning. The framework addresses limitations of traditional graph-based methods by leveraging KAN's spline-based activation functions to explicitly model complex nonlinear spatial-gene interactions through univariate function decompositions. The stkan integrates multi-modal inputs including gene expression, spatial coordinates, and optional morphological features extracted via Vision Transformer. The workflow involves preprocessing with PCA dimensionality reduction, encoding through a KAN-based variational autoencoder combined with graph attention networks, and latent representation fusion enhanced by contrastive learning. Spatial domains are identified using Leiden clustering on the refined embeddings, optimized through a composite loss function combining reconstruction, binary cross-entropy, KL-divergence, and contrastive losses. The stkan demonstrates superior performance across multiple datasets including DLPFC, mouse embryo, and cancer tissues, achieving higher accuracy in domain detection and enabling robust downstream analyses like UMAP visualization, PAGA trajectory inference, and functional enrichment studies.

## install stkan

```python
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch_sparse==0.6.18+pt21cu121 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install stkan
# pip install numpy==1.26.4
```
