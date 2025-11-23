import numpy, torch, networkx
import scipy.sparse as sp
from torch_sparse import SparseTensor

class graph():
    def __init__(self, 
                 data,
                 k,  
                 distType = 'KDTree',):
        super(graph, self).__init__()
        self.data = data
        self.distType = distType
        self.k = k

    def compute(self):
        if self.distType == "KDTree":
            from sklearn.neighbors import KDTree
            tree = KDTree(self.data)
            _, ind = tree.query(self.data, k=self.k+1)
            indices = ind[:, 1:]
            graphList=[]
            for node_idx in range(self.data.shape[0]):
                for j in numpy.arange(0, indices.shape[1]):
                    graphList.append((node_idx, indices[node_idx][j]))
            print('%.4f neighbors per cell on average.' %(len(graphList)/self.data.shape[0]))
        else: 
            raise ValueError(f"""{self.distType!r} does not support.""")
        return graphList

    def list2dict(self, graphList):
        graphdict, tempdict = {}, {}
        for graph in graphList:
            end1 = graph[0]
            end2 = graph[1]
            tempdict[end1] = ""
            tempdict[end2] = ""
            tmplist = graphdict[end1] if end1 in graphdict else []
            tmplist.append(end2)
            graphdict[end1] = tmplist
        for i in range(self.data.shape[0]):
            if i not in tempdict:
                graphdict[i] = []
        return graphdict

    def adj2sparse(self, adj):
        adj = adj.tocoo().astype(numpy.float32)
        row = torch.from_numpy(adj.row).to(torch.long)
        col = torch.from_numpy(adj.col).to(torch.long)
        values = torch.from_numpy(adj.data)
        adj_sparse = SparseTensor(row=row, col=col, value=values, sparse_sizes=adj.shape)
        adj_sparse_t = adj_sparse.t()
        return adj_sparse_t

    def pre_graph(self, adj_pre):
        adj_pre = sp.coo_matrix(adj_pre)
        adj_pre_ = adj_pre + sp.eye(adj_pre.shape[0])
        rowsum = numpy.array(adj_pre_.sum(1))
        degree_mat_inv_sqrt = sp.diags(numpy.power(rowsum, -0.5).flatten())
        adj_normalized = adj_pre_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        return self.adj2sparse(adj_normalized)

    def main(self):
        graphlist = self.compute()
        graphdict = self.list2dict(graphlist)
        adj_net = networkx.adjacency_matrix(networkx.from_dict_of_lists(graphdict))
        adj_pre = adj_net
        adj_pre = adj_pre - sp.dia_matrix((adj_pre.diagonal()[numpy.newaxis, :], [0]), shape=adj_pre.shape)
        adj_pre.eliminate_zeros()
        adj_sparse = self.pre_graph(adj_pre)
        adj_label = adj_pre + sp.eye(adj_pre.shape[0])
        adj_label = torch.FloatTensor(adj_label.toarray())
        norm = adj_pre.shape[0] * adj_pre.shape[0] / float((adj_pre.shape[0] * adj_pre.shape[0] - adj_pre.sum()) * 2)
        graph_dict = {"adj_norm": adj_sparse,
                      "adj_label": adj_label,
                      "norm_value": norm }
        return graph_dict

def combine_graph_dict(dict_1, dict_2):
    tmp_adj_norm = torch.block_diag(dict_1['adj_norm'].to_dense(), dict_2['adj_norm'].to_dense())
    graph_dict = {"adj_norm": SparseTensor.from_dense(tmp_adj_norm),
                  "adj_label": torch.block_diag(dict_1['adj_label'], dict_2['adj_label']),
                  "norm_value": numpy.mean([dict_1['norm_value'], dict_2['norm_value']])}
    return graph_dict