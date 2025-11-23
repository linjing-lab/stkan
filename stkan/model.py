import torch
from torch_geometric.nn import Sequential, BatchNorm
from .kan import KANLinear
from typing import Optional

class vae_kan(torch.nn.Module):
    def __init__(self, 
                input_dim: int,
                Conv_type: str = 'GATConv',
                linear_encoder_hidden: list = [50, 32],
                linear_decoder_hidden: list = [50, 20],
                conv_hidden: list = [50, 12],
                p_drop: float = 0.01,
                dec_cluster_n: int = 15,
                alpha: float = 0.9,
                activate: str = "relu"):
        """
        :param input_dim: int, Dimension of input features (number of genes).
        :param Conv_type: str, Type of graph convolutional layer. default: str='GATConv'. or 'GCNConv'.
        :param linear_encoder_hidden: list, List of hidden layer sizes for the encoder. default: List[int]=[50, 32].
        :param linear_decoder_hidden: list, List of hidden layer sizes for the decoder. default: List[int]=[50, 20].
        :param conv_hidden: list, List of hidden layer sizes for the graph convolutional layers. default: List[int]=[50, 12].
        :param p_drop: float, Dropout probability. default: float=0.01.
        :param dec_cluster_n: int, Number of clusters for DEC. default: int=15.
        :param alpha: float, Parameter for student's t-distribution. default: float=0.9.
        :param activate: str, Activation function. default: str='relu'. or 'sigmoid'.
        """
        super(vae_kan, self).__init__()
        self.input_dim = input_dim
        self.Conv_type = Conv_type
        self.alpha = alpha
        self.conv_hidden = conv_hidden
        self.linear_encoder_hidden = linear_encoder_hidden
        self.linear_decoder_hidden = linear_decoder_hidden
        self.activate = activate
        self.p_drop = p_drop
        self.dec_cluster_n = dec_cluster_n
        temp_dim = self.input_dim
        self.encoder = torch.nn.Sequential()
        for i, hidden_size in enumerate(linear_encoder_hidden):
            self.encoder.add_module(
                f'encoder_L{i}', 
                self._network(temp_dim, hidden_size)
            )
            temp_dim = hidden_size
        temp_dim = linear_encoder_hidden[-1] + conv_hidden[-1]
        self.decoder = torch.nn.Sequential()
        for i, hidden_size in enumerate(linear_decoder_hidden):
            self.decoder.add_module(
                f'decoder_L{i}',
                self._network(temp_dim, hidden_size)
            )
            temp_dim = hidden_size
        self.decoder.add_module(
            'decoder_out',
            KANLinear(temp_dim, self.input_dim)
        )
        self._graph_conv_layers()
        self.dc = InnerProductDecoder(p_drop)
        self.cluster_layer = torch.nn.parameter.Parameter(torch.Tensor(
            self.dec_cluster_n, 
            self.linear_encoder_hidden[-1] + self.conv_hidden[-1]
        ))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def _network(self, in_features: int, out_features: int) -> torch.nn.Sequential:
        """Build a block network with KAN(Kolmogorov-Arnold Network), activation and dropout"""
        layers = [
            KANLinear(in_features, out_features),
            BatchNorm(out_features, momentum=0.01, eps=0.001),
        ]
        if self.activate == "relu":
            layers.append(torch.nn.ELU())
        elif self.activate == "sigmoid":
            layers.append(torch.nn.Sigmoid())
        if self.p_drop > 0:
            layers.append(torch.nn.Dropout(self.p_drop))
        return torch.nn.Sequential(*layers)

    def _graph_conv_layers(self):
        conv_class = self._get_conv_class()
        self.conv = Sequential('x, edge_index', [
            (conv_class(self.linear_encoder_hidden[-1], self.conv_hidden[0]*2), 
            'x, edge_index -> x'),
            BatchNorm(self.conv_hidden[0]*2),
            torch.nn.ReLU(inplace=True), 
        ])
        self.conv_mu = Sequential('x, edge_index', [
            (conv_class(self.conv_hidden[0]*2, self.conv_hidden[-1]), 
            'x, edge_index -> x')
        ])
        self.conv_logvar = Sequential('x, edge_index', [
            (conv_class(self.conv_hidden[0]*2, self.conv_hidden[-1]), 
            'x, edge_index -> x')
        ])
    def _get_conv_class(self):
        """Get the appropriate graph convolution class"""
        conv_classes = {
            "GCNConv": self._import_conv_class("GCNConv"),
            "GATConv": self._import_conv_class("GATConv"),
        }
        return conv_classes[self.Conv_type]
    
    def _import_conv_class(self, class_name: str):
        from torch_geometric.nn import __dict__ as geom_nn_dict
        return geom_nn_dict[class_name]

    def encode(self, x: torch.Tensor, adj: torch.Tensor) -> tuple:
        """
        :param x: torch.Tensor, Input feature matrix [n_nodes, input_dim].
        :param adj: torch.Tensor, Adjacency matrix or edge index [2, n_edges].
            
        :return: tuple: (mu, logvar, feat_x)
            mu: torch.Tensor, Mean of latent distribution [n_nodes, conv_hidden[-1]].
            logvar: torch.Tensor, Log variance of latent distribution [n_nodes, conv_hidden[-1]].
            feat_x: torch.Tensor, Encoded features [n_nodes, linear_encoder_hidden[-1]].
        """
        feat_x = self.encoder(x)
        conv_x = self.conv(feat_x, adj)
        return self.conv_mu(conv_x, adj), self.conv_logvar(conv_x, adj), feat_x

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        :param mu: torch.Tensor, Mean of latent distribution.
        :param logvar: torch.Tensor, Log variance of latent distribution
            
        :return: torch.Tensor, Sampled latent variables.
        """
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        return mu

    def target_distribution(self, target: torch.Tensor) -> torch.Tensor:
        """
        :param target: torch.Tensor, Current soft cluster assignments
            
        :return: torch.Tensor, Target distribution
        """
        weight = (target ** 2) / torch.sum(target, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def stKAN_loss(self, 
                   decoded: torch.Tensor, 
                   x: torch.Tensor, 
                   preds: torch.Tensor, 
                   labels: torch.Tensor, 
                   mu: torch.Tensor, 
                   logvar: torch.Tensor, 
                   n_nodes: int, 
                   norm: float, 
                   mask: Optional[torch.Tensor] = None, 
                   mse_weight: float = 10, 
                   bce_kld_weight: float = 0.1) -> torch.Tensor:
        """
        :param decoded: torch.Tensor, Decoded or reconstructed features.
        :param x: torch.Tensor, Original input features.
        :param preds: torch.Tensor, Predicted adjacency matrix.
        :param labels: torch.Tensor, True adjacency matrix.
        :param mu: torch.Tensor, Mean of latent distribution.
        :param logvar: torch.Tensor, Log variance of latent distribution
        :param n_nodes: int, Number of nodes in graph
        :param norm: float, Normalization factor
        :param mask: torch.Tensor, Mask for adjacency matrix. optional.
        :param mse_weight: float. Weight for reconstruction loss.
        :param bce_kld_weight: float, Weight for BCE and KLD losses.
            
        :return: torch.Tensor, Combined loss value.
        """
        mse_loss = torch.nn.functional.mse_loss(decoded, x)
        if mask is not None:
            preds = preds * mask
            labels = labels * mask
        bce_loss = norm * torch.nn.functional.binary_cross_entropy_with_logits(preds, labels)
        KLD = -0.5 / n_nodes * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        return mse_weight * mse_loss + bce_kld_weight * (bce_loss + KLD)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> tuple:
        """
        :param x: torch.Tensor, Input feature matrix [n_nodes, input_dim].
        :param adj: torch.Tensor, Adjacency matrix or edge index [2, n_edges].
            
        :return: tuple: (z, mu, logvar, de_feat, q, feat_x, graph_z)
            z: torch.Tensor, Combined latent features [n_nodes, linear_encoder_hidden[-1] + conv_hidden[-1]].
            mu: torch.Tensor, Mean of latent distribution.
            logvar: torch.Tensor, Log variance of latent distribution.
            de_feat: torch.Tensor, Decoded features [n_nodes, input_dim].
            q: torch.Tensor, Soft cluster assignments [n_nodes, dec_cluster_n].
            feat_x: torch.Tensor, Encoded features [n_nodes, linear_encoder_hidden[-1]].
            graph_z: torch.Tensor, Graph latent features [n_nodes, conv_hidden[-1]].
        """
        mu, logvar, feat_x = self.encode(x, adj)
        graph_z = self.reparameterize(mu, logvar)
        z = torch.cat((feat_x, graph_z), 1)
        de_feat = self.decoder(z)
        # Compute soft cluster assignments
        q = 1.0 / ((1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha) + 1e-8)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return z, mu, logvar, de_feat, q, feat_x, graph_z


class InnerProductDecoder(torch.nn.Module):
    """
    :param dropout: float, Dropout probability.
    :param act: callable, Activation function. default: torch.sigmoid.
    """
    def __init__(self, dropout: float, act: callable = torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        :param z: torch.Tensor, Latent features [n_nodes, n_features].
            
        :return: torch.Tensor, Reconstructed adjacency matrix [n_nodes, n_nodes].
        """
        z = torch.nn.functional.dropout(z, self.dropout, training=self.training)
        return self.act(torch.mm(z, z.t()))


class GradientReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: float) -> torch.Tensor:
        """
        :param x: torch.Tensor, Input tensor.
        :param weight: float, Weight for gradient scaling.
            
        :return: torch.Tensor, Same as input tensor.
        """
        ctx.weight = weight
        return x.view_as(x) * 1.0

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """
        :param grad_output: torch.Tensor, Gradient from subsequent layers.
            
        :return: tuple: (rev_grad, None)
            rev_grad: torch.Tensor, Reversed and scaled gradients.
            None: Placeholder for weight gradient.
        """
        return (grad_output * -1 * ctx.weight), None


class adv_kan(torch.nn.Module):
    """
    :param model: vae_kan, The base stKAN model of KAN.
    :param n_domains: int, Number of domains to adapt between. default: int=2.
    :param weight: float, Weight for gradient reversal. default: int=1.
    :param n_layers: int, Number of hidden layers. default: int=2.
    """
    def __init__(self,
                 model: vae_kan,
                 n_domains: int = 2,
                 weight: float = 1,
                 n_layers: int = 2):
        super(adv_kan, self).__init__()
        self.model = model
        self.n_domains = n_domains
        self.weight = weight
        self.n_layers = n_layers
        input_dim = self.model.linear_encoder_hidden[-1] + self.model.conv_hidden[-1]
        hidden_layers = []
        for _ in range(n_layers):
            hidden_layers.extend([
                KANLinear(input_dim, input_dim),
                torch.nn.ReLU(),
            ])
        self.domain_clf = torch.nn.Sequential(
            *hidden_layers,
            KANLinear(input_dim, self.n_domains),
        )

    def set_rev_grad_weight(self, weight: float) -> None:
        self.weight = weight

    def target_distribution(
        self, 
        target
        ):
        weight = (target ** 2) / torch.sum(target, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def stKAN_loss(self, 
                   decoded: torch.Tensor, 
                   x: torch.Tensor, 
                   preds: torch.Tensor, 
                   labels: torch.Tensor, 
                   mu: torch.Tensor, 
                   logvar: torch.Tensor, 
                   n_nodes: int, 
                   norm: float, 
                   mask: Optional[torch.Tensor] = None, 
                   mse_weight: float = 10, 
                   bce_kld_weight: float = 0.1) -> torch.Tensor:
        """
        :param decoded: torch.Tensor, Decoded or reconstructed features.
        :param x: torch.Tensor, Original input features.
        :param preds: torch.Tensor, Predicted adjacency matrix.
        :param labels: torch.Tensor, True adjacency matrix.
        :param mu: torch.Tensor, Mean of latent distribution.
        :param logvar: torch.Tensor, Log variance of latent distribution.
        :param n_nodes: int, Number of nodes in graph.
        :param norm: float, Normalization factor.
        :param mask: torch.Tensor, Mask for adjacency matrix. optional.
        :param mse_weight: float, Weight for reconstruction loss.
        :param bce_kld_weight: float, Weight for BCE and KLD losses.
            
        :return torch.Tensor, Combined loss value.
        """
        mse_loss = torch.nn.functional.mse_loss(decoded, x)
        if mask is not None:
            preds = preds * mask
            labels = labels * mask
        bce_loss = norm * torch.nn.functional.binary_cross_entropy_with_logits(preds, labels)
        KLD = -0.5 / n_nodes * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        return mse_weight * mse_loss + bce_kld_weight * (bce_loss + KLD)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple:
        """
        :param x: torch.Tensor, Input features [n_nodes, input_dim].
        :param edge_index: torch.Tensor, Graph edge indices [2, n_edges].
            
        :return: tuple: (z, mu, logvar, de_feat, q, feat_x, graph_z, domain_pred)
            domain_pred: torch.Tensor, Domain classification logits [n_nodes, n_domains].
        """
        z, mu, logvar, de_feat, q, feat_x, graph_z = self.model(x, edge_index)
        x_reverse = GradientReverseLayer.apply(z, self.weight)
        domain_pred = self.domain_clf(x_reverse)
        return z, mu, logvar, de_feat, q, feat_x, graph_z, domain_pred