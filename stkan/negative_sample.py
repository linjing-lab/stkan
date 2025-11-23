import torch, random, numpy

class nega_sample():
    def __init__(self,  knn_neighbor, device, pool_percent, sample_percent, topkdevice):
        self.knn_neighbor = knn_neighbor
        self.device = device
        self.negapool = None
        self.negasample = None
        self.pool_percent = pool_percent
        self.sample_percent = sample_percent 
        self.topkdevice = topkdevice

    def fit(self,similarity,seed):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        numpy.random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        x = self.knn_neighbor.coalesce().indices()[0]
        y = self.knn_neighbor.coalesce().indices()[1]
        similarity[x, y] = 1.0
        self.pool_size = int(similarity.shape[0] * self.pool_percent)
        self.n_sample = self.pool_size * self.sample_percent
        _,self.negapool = similarity.topk(k=self.pool_size,largest=False,sorted=False)
        sample_ind = torch.tensor(random.sample(range(int(self.negapool.shape[1])),int(self.n_sample))).to(self.topkdevice)
        negasample = torch.index_select(self.negapool.to(self.topkdevice),1,sample_ind)
        self.negasample = self.create_sparse(negasample.cpu())
        self.n_nega = sample_ind.shape[0]

    def create_sparse(self, I):
        similar = I.reshape(-1)
        index = numpy.repeat(range(I.shape[0]), I.shape[1])
        assert len(similar) == len(index)
        indices = torch.tensor(numpy.vstack((index,numpy.array(similar))))
        result = torch.sparse_coo_tensor(indices, torch.ones_like(I.reshape(-1)).cpu(), [I.shape[0], I.shape[0]])
        return result
