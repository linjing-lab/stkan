import os, numpy, scanpy, random, torch, pandas
from typing import Literal
import matplotlib.pyplot as plt

_QUALITY = Literal["fulres", "hires", "lowres"]
_background = ["black", "white"]

def fixed(seed=0):   
    random.seed(seed)    
    os.environ['PYTHONHASHSEED'] = str(seed)    
    numpy.random.seed(seed)    
    torch.manual_seed(seed)    
    torch.cuda.manual_seed(seed)    
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False    
    torch.backends.cudnn.deterministic = True

def read_10X_Visium(path, 
                    genome = None,
                    count_file ='filtered_feature_bc_matrix.h5', 
                    library_id = None, 
                    load_images =True, 
                    quality ='hires',
                    image_path = None
                   ):
    adata = scanpy.read_visium(path, genome = genome, count_file = count_file, library_id = library_id, load_images = load_images,)
    adata.var_names_make_unique()
    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]
    if quality == "fulres":
        image_coor = adata.obsm["spatial"]
        img = plt.imread(image_path, 0)
        adata.uns["spatial"][library_id]["images"]["fulres"] = img
    else:
        scale = adata.uns["spatial"][library_id]["scalefactors"][
            "tissue_" + quality + "_scalef"]
        image_coor = adata.obsm["spatial"] * scale
    adata.obs["imagecol"] = image_coor[:, 0]
    adata.obs["imagerow"] = image_coor[:, 1]
    adata.uns["spatial"][library_id]["use_quality"] = quality
    return adata

def refine(sample_id, pred, dis, shape="hexagon"):
    refined_pred=[]
    pred=pandas.DataFrame({"pred": pred}, index=sample_id)
    dis_df=pandas.DataFrame(dis, index=sample_id, columns=sample_id)
    if shape=="hexagon":
        num_nbs=6 
    elif shape=="square":
        num_nbs=4
    else:
        raise ValueError(f"Unexpected shape type: {shape}. shape='hexagon' for Visium data, 'square' for ST data.")
    for i in range(len(sample_id)):
        index=sample_id[i]
        dis_tmp=dis_df.loc[index, :].sort_values()
        nbs=dis_tmp[0:num_nbs+1]
        nbs_pred=pred.loc[nbs.index, "pred"]
        self_pred=pred.loc[index, "pred"]
        v_c=nbs_pred.value_counts()
        if (v_c.loc[self_pred]<num_nbs/2) and (numpy.max(v_c)>num_nbs/2):
            refined_pred.append(v_c.idxmax())
        else:           
            refined_pred.append(self_pred)
    return refined_pred

def contrastive_loss(z, positive_emb, mask_nega, n_nega,device, temperature=0.5):
    emb = torch.nn.functional.normalize(z, dim=-1, p=2)
    similarity = torch.matmul(emb, torch.transpose(emb, 1, 0).detach())
    e_sim = torch.exp((similarity / temperature))
    positive_emb_norm = torch.nn.functional.normalize(positive_emb, dim=-1, p=2).to(device)
    positive_sim = torch.exp((positive_emb_norm * emb.unsqueeze(1)).sum(axis=-1) / temperature)
    x = mask_nega._indices()[0]
    y = mask_nega._indices()[1]
    N_each_spot = e_sim[x,y].reshape((-1,n_nega)).sum(dim=-1)
    N_each_spot = N_each_spot.unsqueeze(-1).repeat([1,positive_sim.shape[1]])
    contras = -torch.log(positive_sim / (positive_sim + N_each_spot))
    return torch.mean(contras)