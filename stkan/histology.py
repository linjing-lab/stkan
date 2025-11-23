import os, numpy, pandas, random, torch, torchvision
from PIL import Image
from pathlib import Path
from sklearn.decomposition import PCA
from tqdm import tqdm

class image_feature:
    def __init__(
        self,
        adata,
        pca_components=50,
        transType='ViT16',
        seeds=0,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.adata = adata
        self.pca_components = pca_components
        self.seeds = seeds
        self.transType = transType

    def load_trans_model(self):
        vit_map = {
            'ViT16': (torchvision.models.vit_b_16, torchvision.models.ViT_B_16_Weights.DEFAULT),
        }
        if self.transType not in vit_map:
            raise ValueError(f"{self.transType} is not a valid ViT type. Options: {list(vit_map.keys())}")
        vit_model, pretrain_weight = vit_map[self.transType]
        model = vit_model(weights=pretrain_weight)
        model.to(self.device)
        return model

    def extract(self):
        transform_list = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            torchvision.transforms.RandomAutocontrast(),
            torchvision.transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1.)),
            torchvision.transforms.RandomInvert(),
            torchvision.transforms.RandomAdjustSharpness(random.uniform(0, 1)),
            torchvision.transforms.RandomSolarize(random.uniform(0, 1)),
            torchvision.transforms.RandomAffine(45, translate=(0.3, 0.3), scale=(0.8, 1.2), shear=(-0.3, 0.3, -0.3, 0.3)),
            torchvision.transforms.RandomErasing()
        ]
        image2tensor = torchvision.transforms.Compose(transform_list)
        features_container, spots = [], []
        model = self.load_trans_model()
        model.eval()
        if "slices_path" not in self.adata.obs.keys():
            raise ValueError("Please execute image_crop() first to generate spot images")
        with tqdm(total=len(self.adata),
                  desc="Extracting image features",
                  bar_format="{l_bar}{bar} [ time left: {remaining} ]", ncols=80, dynamic_ncols=True, leave=True) as pbar:
            for spot, path in self.adata.obs['slices_path'].items():
                slice = Image.open(path).resize((224, 224))
                slice = numpy.asarray(slice, dtype=numpy.float32)
                cur_tensor = image2tensor(slice).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    features = model(torch.autograd.Variable(cur_tensor)).cpu().numpy().ravel()
                features_container.append(features)
                spots.append(spot)
                pbar.update(1)
        feat_frame = pandas.DataFrame(features_container, index=spots)
        self.adata.obsm["image_features"] = feat_frame.to_numpy()
        pca = PCA(n_components=self.pca_components, random_state=self.seeds)
        self.adata.obsm["pca_image_features"] = pca.fit_transform(self.adata.obsm["image_features"])
        return self.adata

def image_crop(
    adata,
    save_path,
    library_id=None,
    crop_size=50,
    target_size=224,
):
    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]
    image = adata.uns["spatial"][library_id]["images"][
        adata.uns["spatial"][library_id]["use_quality"]]
    if image.dtype == numpy.float32 or image.dtype == numpy.float64:
        image = (image * 255).astype(numpy.uint8)
    img_pillow = Image.fromarray(image)
    os.makedirs(save_path, exist_ok=True)
    tile_names = []
    with tqdm(total=len(adata),
              desc="Cropping spot images",
              bar_format="{l_bar}{bar} [ time left: {remaining} ]", ncols=80, dynamic_ncols=True, leave=True) as pbar:
        for imagerow, imagecol in zip(adata.obs["imagerow"], adata.obs["imagecol"]):
            left, upper, right, lower = imagecol - crop_size / 2, imagerow - crop_size / 2, imagecol + crop_size / 2, imagerow + crop_size / 2
            tile = img_pillow.crop((left, upper, right, lower))
            tile = tile.resize((target_size, target_size))
            tile_name = f"{imagecol}-{imagerow}-{crop_size}.png"
            out_path = Path(save_path) / tile_name
            tile.save(out_path, "PNG")
            tile_names.append(str(out_path))
            pbar.update(1)
    adata.obs["slices_path"] = tile_names
    return adata
