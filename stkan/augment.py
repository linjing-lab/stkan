import numpy, pandas
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.neighbors import BallTree

def cal_spatial_weight(
    data,
    spatial_k=50,
    spatial_type="BallTree",
):
    """
    Calculate binary spatial weight matrix based on BallTree.
    :param data: numpy.ndarray, Spatial coordinates array of shape (n_spots, 2).
    :param spatial_k: int, Number of nearest neighbors to consider. default: int=50.
    :param spatial_type: str, Algorithm for neighbor search. default: str="BallTree".
    
    :return: numpy.ndarray, Binary spatial weight matrix of shape (n_spots, n_spots).
    """
    if spatial_type == "BallTree":
        tree = BallTree(data, leaf_size=2)
        _, inds = tree.query(data, k=spatial_k+1)
    inds = inds[:, 1:]
    spot2spot = numpy.zeros((data.shape[0], data.shape[0]))
    for i in range(inds.shape[0]):
        spot2spot[i, inds[i]] = 1
    return spot2spot

def cal_gene_weight(
    data,
    n_components=50,
    gene_dist_type="cosine",
):
    """
    :param data: numpy.ndarray or scipy.sparse.csr_matrix, Gene expression matrix of shape (n_spots, n_genes).
    :param n_components: int, Number of PCA components if n_genes > 500. default: int=50.
    :param gene_dist_type: str, Distance metric for calculating gene similarity. default: str="cosine".
        
    :return: numpy.ndarray, Gene expression similarity matrix of shape (n_spots, n_spots).
    """
    if isinstance(data, csr_matrix):
        data = data.toarray()
    if data.shape[1] > 500:
        pca = PCA(n_components=n_components)
        data = pca.fit_transform(data)
    return 1 - pairwise_distances(data, metric=gene_dist_type)

def cal_weight_matrix(
    adata,
    md_dist_type="cosine",
    gb_dist_type="correlation",
    n_components=50,
    use_morphological=True,
    spatial_k=30,
    spatial_type="BallTree",
    verbose=False,
):
    """
    :param adata: anndata.AnnData, Spatial transcriptomics dataset containing:
        - obsm['spatial']: Spatial coordinates
        - X: Gene expression matrix
        - obsm['image_feat_pca']: Morphological features (if use_morphological=True)
    :param md_dist_type: str, Distance metric for morphological similarity. default: str="cosine".
    :param gb_dist_type: str, Distance metric for gene expression similarity. default: str="correlation".
    :param n_components: int, Number of PCA components for gene expression. default: int=50.
    :param use_morphological: bool, Whether to include morphological similarity. default: bool=True.
    :param spatial_k: int, Number of spatial neighbors to consider. default: int=30.
    :param spatial_type: str, Method for spatial neighbor calculation. default: str="BallTree".
    :param verbose: bool, Whether to store intermediate matrices in adata.obsm. default: bool=False.
        
    :return: anndata.AnnData,
        Updated AnnData object with:
        - obsm['weight_matrix']: Combined weight matrix
        - obsm['gene_spot']: Gene weights (if verbose=True)
        - obsm['spot2spot']: Spatial weights (if verbose=True)
        - obsm['morpho_weights']: Morphological weights (if verbose=True and use_morphological=True)
    """
    spot2spot = cal_spatial_weight(
        adata.obsm['spatial'], 
        spatial_k=spatial_k, 
        spatial_type=spatial_type
    )
    print(f"Spatial weights calculated. Average neighbors: {spot2spot.sum()/adata.shape[0]:.1f}")
    gene_spot = cal_gene_weight(
        data=adata.X.copy(),
        gene_dist_type=gb_dist_type,
        n_components=n_components
    )
    print("Gene expression weights calculated.")
    if verbose:
        adata.obsm["gene_spot"] = gene_spot
        adata.obsm["spot2spot"] = spot2spot
    if use_morphological:
        morpho_weights = 1 - pairwise_distances(
            numpy.array(adata.obsm["image_feat_pca"]), 
            metric=md_dist_type
        )
        morpho_weights[morpho_weights < 0] = 0
        print("Morphological weights calculated.")
        if verbose:
            adata.obsm["morpho_weights"] = morpho_weights
        adata.obsm["weight_matrix"] = (
            spot2spot * 
            gene_spot * 
            morpho_weights
        )
    else:
        adata.obsm["weight_matrix"] = (
            gene_spot * 
            spot2spot
        )
    print("Final weight matrix calculated and stored in adata.obsm['weight_matrix']")
    return adata

def find_adjacent_spot(
    adata,
    use_data="raw",
    neighbour_k=4,
    verbose=False,
):
    """ 
    :param adata: anndata.AnnData, Dataset containing weight_matrix in obsm
    :param use_data: str, Data source to use. default: str="raw".
    :param neighbour_k: int, Number of top neighbors to consider. default: int=4.
    :param verbose: bool, Whether to store neighbor weights in adata.obsm. default: bool=False.
        
    :return: anndata.AnnData, Updated AnnData with:
        - obsm['adjacent_data']: Weighted neighbor gene expression
        - obsm['adjacent_weight']: Neighbor weights (if verbose=True)
    """
    # Get expression data from specified source
    if use_data == "raw":
        if isinstance(adata.X, csr_matrix):
            adataX = adata.X.toarray()
        elif isinstance(adata.X, numpy.ndarray):
            adataX = adata.X
        elif isinstance(adata.X, pandas.DataFrame):
            adataX = adata.X.values
        else:
            raise ValueError(f"Unsupported data type: {type(adata.X)}")
    else:
        adataX = adata.obsm[use_data]
    weights, coordinates = [], []
    with tqdm(total=len(adata), desc="Finding adjacent spots",
              bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
        for i in range(adata.shape[0]):
            spot = adata.obsm['weight_matrix'][i].argsort()[-neighbour_k:][:neighbour_k-1]
            weight = adata.obsm['weight_matrix'][i][spot]
            cur_matrix = adataX[spot]
            if weight.sum() > 0:
                weight_scaled = weight / weight.sum()
                weights.append(weight_scaled)
                cur_matrix_scaled = weight_scaled.reshape(-1,1) * cur_matrix
                cur_matrix_final = numpy.sum(cur_matrix_scaled, axis=0)
            else:
                cur_matrix_final = numpy.zeros(adataX.shape[1])
                weights.append(numpy.zeros(len(spot)))
            coordinates.append(cur_matrix_final)
            pbar.update(1)
    adata.obsm['adjacent_data'] = numpy.array(coordinates)
    if verbose:
        adata.obsm['adjacent_weight'] = numpy.array(weights)
    return adata


def augment_gene_data(
    adata,
    adjacent_weight=0.2,
):
    """
    :param adata: anndata.AnnData, Dataset containing:
        - X: Original gene expression
        - obsm['adjacent_data']: Neighbor contributions
    :param adjacent_weight: float, Weight for neighbor contribution. default: float=0.2.
        
    :return: anndata.AnnData, Updated AnnData with augmented data in obsm['augment_gene_data'].
    """
    if isinstance(adata.X, numpy.ndarray):
        augmented_matrix = adata.X + adjacent_weight * adata.obsm["adjacent_data"].astype(float)
    elif isinstance(adata.X, csr_matrix):
        augmented_matrix = adata.X.toarray() + adjacent_weight * adata.obsm["adjacent_data"].astype(float)
    adata.obsm["augment_gene_data"] = augmented_matrix
    return adata


def augment_adata(
    adata,
    md_dist_type="cosine",
    gb_dist_type="correlation",
    n_components=50,
    use_morphological=True,
    use_data="raw",
    neighbour_k=4,
    adjacent_weight=0.2,
    spatial_k=30,
    spatial_type="BallTree",
):
    """
    :param adata: anndata.AnnData, Input spatial transcriptomics data.
    :param md_dist_type: str, Morphological distance metric. default: str="cosine".
    :param gb_dist_type: str, Gene expression distance metric. default: str="correlation".
    :param n_components: int, PCA components for gene expression. default: int=50.
    :param use_morphological: bool, Whether to use morphological features. default: bool=True.
    :param use_data: str, Data source for expression. default: str="raw".
    :param neighbour_k: int, Number of neighbors to consider. default: int=4.
    :param adjacent_weight: float, Weight for neighbor contributions. default: float=0.2.
    :param spatial_k: int, Spatial neighbors to consider. default: int=30.
    :param spatial_type: Spatial neighbor algorithm. default: str="BallTree".
        
    :return: anndata.AnnData, Augmented dataset with:
        - obsm['weight_matrix']: Combined weights
        - obsm['adjacent_data']: Neighbor contributions
        - obsm['augment_gene_data']: Final augmented data
    """
    return augment_gene_data(
        find_adjacent_spot(
        cal_weight_matrix(
        adata,
        md_dist_type=md_dist_type,
        gb_dist_type=gb_dist_type,
        n_components=n_components,
        use_morphological=use_morphological,
        spatial_k=spatial_k,
        spatial_type=spatial_type,
    ),
        use_data=use_data,
        neighbour_k=neighbour_k,
    ),
        adjacent_weight=adjacent_weight,
    )