import os, time, psutil, numpy, pandas, scanpy, anndata
from sklearn.metrics import calinski_harabasz_score
from scipy.spatial import distance
from typing import Optional, List
from .utils import read_10X_Visium, refine
from .adjacent import combine_graph_dict # Integration
from .model import vae_kan, adv_kan
from .train_config import train


class executor():
    def __init__(
        self,
        task: str = "identify",
        pre_epochs: int = 500,
        epochs: int = 500,
        use_gpu: bool = True
    ):
        self.task = task
        self.pre_epochs = pre_epochs
        self.epochs = epochs
        self.use_gpu = use_gpu

    def get_adata(
        self,
        platform: str,
        data_path: str,
        data_name: str,
    ) -> scanpy.AnnData:
        platforms = ['Visium', 'ST']
        if platform not in platforms:
            raise ValueError(f"Platform must be one of {platforms}")
        read_functions = {
            'Visium': read_10X_Visium,
        }
        if platform != 'ST':
            adata = read_functions[platform](os.path.join(data_path, data_name))
        else:
            raise RuntimeError(f"Read functions not support {platforms}")
        return adata
    
    def get_multiple_adata(
        self,
        adata_list: List[scanpy.AnnData],
        data_name_list: List[str],
        graph_list: List[dict]
    ) -> tuple:
        multiple_adata, multiple_graph = None, None
        for i, (current_adata, current_name) in enumerate(zip(adata_list, data_name_list)):
            current_adata.obs['batch_name'] = current_name
            current_adata.obs['batch_name'] = current_adata.obs['batch_name'].astype('category')
            if i == 0:
                multiple_adata = current_adata
                multiple_graph = graph_list[i]
            else:
                var_names = multiple_adata.var_names.intersection(current_adata.var_names)
                multiple_adata = multiple_adata[:, var_names]
                current_adata = current_adata[:, var_names]
                multiple_adata = anndata.concat([multiple_adata, current_adata])
                multiple_graph = combine_graph_dict(multiple_graph, graph_list[i])
        multiple_adata.obs["batch"] = pandas.Categorical(
            multiple_adata.obs['batch_name']
        ).codes.astype(numpy.int64)
        return multiple_adata, multiple_graph

    def data_process(
        self,
        adata: scanpy.AnnData,
        pca_n_comps: int = 200
    ) -> numpy.ndarray:
        adata.raw = adata
        adata.X = adata.obsm["augment_gene_data"].astype(numpy.float64)
        
        # Normalization pipeline
        data = scanpy.pp.normalize_total(adata, target_sum=1, inplace=False)['X']
        data = scanpy.pp.log1p(data)
        data = scanpy.pp.scale(data)
        data = scanpy.pp.pca(data, n_comps=pca_n_comps)
        return data

    def fit(
        self,
        data: numpy.ndarray,
        graph_dict: dict,
        domains: Optional[numpy.ndarray] = None,
        n_domains: Optional[int] = None,
        conv_type: str = "GATConv",
        linear_encoder_hidden: List[int] = [50, 32],
        linear_decoder_hidden: List[int] = [50, 20],
        conv_hidden: List[int] = [50, 12],
        p_drop: float = 0.01,
        dec_cluster_n: int = 20,
        kl_weight: float = 1,
        mse_weight: float = 1,
        bce_kld_weight: float = 1,
        domain_weight: float = 1,
        use_contra: bool = False
    ) -> numpy.ndarray:
        print("Running stKAN analysis...")
        start_time = time.time()
        
        # Initialize model
        model = vae_kan(input_dim=data.shape[1],
                        Conv_type=conv_type,
                        linear_encoder_hidden=linear_encoder_hidden,
                        linear_decoder_hidden=linear_decoder_hidden,
                        conv_hidden=conv_hidden,
                        p_drop=p_drop,
                        dec_cluster_n=dec_cluster_n,)
        
        # Configure for task type
        if self.task == "identify":
            trainer = train(
                data, graph_dict, model,
                pre_epochs=self.pre_epochs,
                epochs=self.epochs,
                kl_weight=kl_weight,
                mse_weight=mse_weight,
                bce_kld_weight=bce_kld_weight,
                domain_weight=domain_weight,
                use_gpu=self.use_gpu,
                use_contra=use_contra
            )
        elif self.task == "integration":
            if n_domains is None:
                raise ValueError("n_domains must be specified for integration task, such as len(SAMPLE_IDS).")
            adv_model = adv_kan(model=model, n_domains=n_domains)
            trainer = train(
                data, graph_dict, adv_model,
                domains=domains,
                pre_epochs=self.pre_epochs,
                epochs=self.epochs,
                kl_weight=kl_weight,
                mse_weight=mse_weight,
                bce_kld_weight=bce_kld_weight,
                domain_weight=domain_weight,
                use_gpu=self.use_gpu,
                use_contra=use_contra
            )
        else:
            raise ValueError(f"Unexpected task type: {self.task}. Support 'identify' and 'integration'.")
        
        # Run training
        trainer.fit()
        embeddings, _ = trainer.process()
        
        # Print stats
        mem_usage = psutil.Process(os.getpid()).memory_info().rss / 1024**3
        total_time = (time.time() - start_time) / 60
        print(f"stKAN training completed")
        print(f"Memory usage: {mem_usage:.2f} GB")
        print(f"Total time: {total_time:.2f} minutes")
        return embeddings

    def get_cluster_data(
        self,
        adata: scanpy.AnnData,
        n_domains: int,
        priori: bool = True,
        batch_key: Optional[str] = None,
        use_obsm: str = "stKAN_embed",
        key_added: str = "stKAN_domain",
        output_key: str = "stKAN_refine_domain",
        shape: str = "hexagon",  # or "square"
    ) -> scanpy.AnnData:
        """
        :param adata: scanpy.AnnData, Annotated data matrix. Requires:
              - `adata.obsm[use_obsm]`: embedding representation for clustering
              - `adata.obsm['spatial']`: spatial coordinates of each spot/cell
        :param n_domains: int, Target number of spatial domains (clusters).
        :param priori: bool, If True, use `n_domains` directly. If False, optimize number of clusters automatically.
        :param batch_key: str or None, Key in `adata.obs` that identifies distinct tissue slices. If None, assumes single-slice data.
        :param use_obsm: str, Name of the key in `adata.obsm` containing embedding to use for clustering.
        :param key_added: str, Name of the column in `adata.obs` where Leiden clustering results will be stored.
        :param output_key: str, Name of the column in `adata.obs` to store refined cluster labels based on spatial neighbors.
        :param shape: str, Neighborhood structure to use during refinement. Choose "hexagon" for Visium-like data or "square" for grid-based ST platforms.

        :return adata: scanpy.AnnData, The updated AnnData object with:
              - `adata.obs[key_added]`: initial clustering labels from Leiden
              - `adata.obs[output_key]`: refined domain labels (post-spatial-smoothing)
        """
        scanpy.pp.neighbors(adata, use_rep=use_obsm)
        if priori:
            res = self._priori_cluster(adata, n_domains)
        else:
            res = self._optimize_cluster(adata)
        scanpy.tl.leiden(adata, key_added=key_added, resolution=res, flavor="igraph", n_iterations=2, directed=False)
        if batch_key and batch_key in adata.obs.columns:
            result = []
            for b in adata.obs[batch_key].unique():
                sub = adata[adata.obs[batch_key] == b]
                adj_2d = distance.cdist(sub.obsm['spatial'], sub.obsm['spatial'], 'euclidean')
                refined = refine(sub.obs_names.tolist(), sub.obs[key_added].tolist(), adj_2d, shape)
                result.extend(zip(sub.obs_names.tolist(), refined))
            adata.obs[output_key] = pandas.Series(dict(result))
        else:
            adj_2d = distance.cdist(adata.obsm['spatial'], adata.obsm['spatial'], 'euclidean')
            refined = refine(adata.obs_names.tolist(), adata.obs[key_added].tolist(), adj_2d, shape)
            adata.obs[output_key] = refined
        return adata
    
    def _optimize_cluster(
        self,
        adata: scanpy.AnnData,
        resolution_range: List[float] = None
    ) -> float:
        if resolution_range is None:
            resolution_range = list(np.arange(0.1, 2.5, 0.01))
        scores = []
        for r in resolution_range:
            scanpy.tl.leiden(adata, resolution=r, flavor="igraph", n_iterations=2, directed=False)
            s = calinski_harabasz_score(adata.X, adata.obs["leiden"])
            scores.append(s)    
        best_idx = numpy.argmax(scores)
        best_res = resolution_range[best_idx]
        print(f"Optimal resolution: {best_res:.2f}")
        return best_res

    def _priori_cluster(
        self,
        adata: scanpy.AnnData,
        n_domains: int
    ) -> float:
        for res in sorted(numpy.arange(0.1, 2.5, 0.01), reverse=True):
            scanpy.tl.leiden(adata, random_state=0, resolution=res, flavor="igraph", n_iterations=2, directed=False)
            if len(adata.obs['leiden'].unique()) == n_domains:
                print(f"Found resolution: {res:.2f} for {n_domains} domains")
                return res
        return 1.0
