import numpy, torch, faiss, random
class build_pair():
    def __init__(self, device, num_centroids, clus_num_iters):
        self.device = device
        self.num_centroids = num_centroids
        self.clus_num_iters = clus_num_iters

    def create_sparse(self, I):
        similar = I.reshape(-1).tolist()
        index = numpy.repeat(range(I.shape[0]), I.shape[1])
        assert len(similar) == len(index)
        indices = torch.tensor(numpy.vstack((index, numpy.array(similar)))).to(self.device)
        result = torch.sparse_coo_tensor(indices, torch.ones_like(I.reshape(-1)), [I.shape[0], I.shape[0]])
        return result

    def cal_kmeans_centroid(self, emb, I_kmenas):
        centroids = []
        for i in range(self.num_centroids):
            ind_tmp = numpy.where(I_kmenas == i)[0] 
            cen_tmp = emb[ind_tmp,:].mean(axis=0) 
            centroids.append(cen_tmp)
        return numpy.array(centroids)

    def cal_positive_emb(self, emb, I_knn, I_kmeans, centroids):
        I_knn = I_knn.cpu()
        spot_centroid = numpy.apply_along_axis(lambda x: centroids[x], 0, I_kmeans).squeeze()
        spot_candi_emb = emb[I_knn.reshape(-1,1).squeeze().numpy(),:].reshape((-1,I_knn.shape[1],emb.shape[1]))
        expand_centroid = numpy.expand_dims(spot_centroid,axis=1).repeat(I_knn.shape[1],axis=1) 
        euc_distance = numpy.exp(numpy.sqrt(numpy.power((expand_centroid - spot_candi_emb), 2).sum(axis=2)))
        prob = torch.nn.Softmax(dim=1)(1 / torch.tensor(euc_distance)).unsqueeze(1)
        positive_emb = torch.matmul(prob, torch.tensor(spot_candi_emb))
        return positive_emb

    def fit(self, z, top_k):
        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        numpy.random.seed(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        z = torch.nn.functional.normalize(z, dim=-1, p=2)
        _, d = z.shape
        similarity = torch.matmul(z, torch.transpose(z, 1, 0).detach())
        _, I_knn = similarity.topk(k=top_k, dim=1, largest=True, sorted=True)
        knn_neighbor = self.create_sparse(I_knn)
        ncentroids = self.num_centroids
        niter = self.clus_num_iters
        emb = z.cpu().numpy()
        all_positive_emb = []
        kmeans = faiss.Kmeans(d, ncentroids, niter=niter, gpu=False, seed=0)
        kmeans.train(emb)
        _, I_kmeans = kmeans.index.search(emb, 1)
        centroids = self.cal_kmeans_centroid(emb, I_kmeans) #nparray shape:[n_centroids, embsize]
        positive_emb = self.cal_positive_emb(emb, I_knn, I_kmeans, centroids)
        if len(all_positive_emb) == 0:
            all_positive_emb = positive_emb
        else:
            all_positive_emb = numpy.concatenate((all_positive_emb,positive_emb),axis=1)
        return all_positive_emb, similarity, knn_neighbor
