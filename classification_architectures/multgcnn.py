import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardGCNNLayer(nn.Module):
    """
    A single layer of a Graph Convolutional Network (GCN).

    Parameters
    ----------
    S : torch.Tensor
        Graph shift operator (e.g., adjacency matrix).
    in_dim : int
        Input dimensionality.
    out_dim : int
        Output dimensionality.
    """

    def __init__(self, S: torch.Tensor, in_dim: int, out_dim: int):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.S = S.clone() + torch.eye(S.shape[0], device=S.device)  # Add self-loops
        self.d = self.S.sum(1)
        self.D_inv = torch.diag(1 / torch.sqrt(self.d))
        self.S = self.D_inv @ self.S @ self.D_inv  # Normalize adjacency matrix
        self.S = nn.Parameter(self.S, requires_grad=False)

        self.W = nn.Parameter(torch.empty(in_dim, out_dim))
        nn.init.xavier_uniform_(self.W)

        self.b = nn.Parameter(torch.empty(out_dim))
        nn.init.uniform_(self.b, -1 / (in_dim * out_dim), 1 / (in_dim * out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GCN layer.

        Parameters
        ----------
        x : torch.Tensor
            Input feature matrix of shape (num_nodes, in_dim).

        Returns
        -------
        torch.Tensor
            Output feature matrix of shape (num_nodes, out_dim).
        """
        return self.S @ x @ self.W + self.b[None, :]


class GraphBranch(nn.Module):
    """
    A GNN branch for processing temporal or static data.

    Parameters
    ----------
    n_layers : int
        Number of GNN layers.
    dropout : float
        Dropout rate for regularization.
    hid_dim : int
        Dimensionality of hidden layers.
    S : torch.Tensor
        Graph shift operator.
    in_dim : int
        Dimensionality of input features.
    embedding_dim : int
        Dimensionality of the output embedding.
    seed : int
        Random seed for reproducibility.
    nonlin : nn.Module, optional
        Non-linearity function, by default nn.LeakyReLU.
    """

    def __init__(
        self,
        n_layers: int,
        dropout: float,
        hid_dim: int,
        S: torch.Tensor,
        in_dim: int,
        embedding_dim: int,
        seed: int,
        nonlin: nn.Module = nn.LeakyReLU,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        self.nonlin = nonlin()
        self.dropout = dropout
        self.batch_norms = nn.ModuleList()

        if n_layers > 1:
            self.layers.append(StandardGCNNLayer(S, in_dim, hid_dim))
            self.batch_norms.append(nn.BatchNorm1d(hid_dim))
            for _ in range(n_layers - 2):
                self.layers.append(StandardGCNNLayer(S, hid_dim, hid_dim))
                self.batch_norms.append(nn.BatchNorm1d(hid_dim))
            self.layers.append(StandardGCNNLayer(S, hid_dim, embedding_dim))
        else:
            self.layers.append(StandardGCNNLayer(S, in_dim, embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the GNN branch.

        Parameters
        ----------
        x : torch.Tensor
            Input feature matrix of shape (num_nodes, in_dim).

        Returns
        -------
        torch.Tensor
            Output embedding matrix of shape (num_nodes, embedding_dim).
        """
        for i, layer in enumerate(self.layers[:-1]):
            x = self.nonlin(layer(x))
            # print(x.shape)
            # if len(self.batch_norms) > i:
            #     x = self.batch_norms[i](x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.layers[-1](x)
        return x


class MultimodalGraphNet(nn.Module):
    """
    Multimodal architecture for processing temporal and static data using GNNs.

    Parameters
    ----------
    n_layers : int
        Number of GNN layers.
    dropout : float
        Dropout rate for regularization.
    hid_dim : int
        Dimensionality of hidden layers.
    S_temporal : torch.Tensor
        Graph shift operator for temporal data.
    S_static : torch.Tensor
        Graph shift operator for static data.
    in_dim_temporal : int
        Dimensionality of input features for temporal data.
    in_dim_static : int
        Dimensionality of input features for static data.
    embedding_dim_static : int
        Dimensionality of the static embedding.
    embedding_dim_temporal : int
        Dimensionality of the temporal embedding.
    fc_out_dim : int
        Dimensionality of the output layer in the fully connected network.
    seed : int
        Random seed for reproducibility.
    nonlin : nn.Module, optional
        Non-linearity function, by default nn.LeakyReLU.
    """

    def __init__(
        self,
        n_layers: int,
        dropout: float,
        hid_dim: int,
        S_temporal: torch.Tensor,
        S_static: torch.Tensor,
        in_dim_temporal: int,
        in_dim_static: int,
        embedding_dim_static: int,
        embedding_dim_temporal: int,
        fc_out_dim: int,
        seed: int,
        nonlin: nn.Module = nn.LeakyReLU,
    ):
        super().__init__()

        self.temporal_branch = GraphBranch(
            n_layers=n_layers,
            dropout=dropout,
            hid_dim=hid_dim,
            S=S_temporal,
            in_dim=in_dim_temporal,
            embedding_dim=1,
            seed=seed,
            nonlin=nonlin,
        )

        self.static_branch = GraphBranch(
            n_layers=n_layers,
            dropout=dropout,
            hid_dim=hid_dim,
            S=S_static,
            in_dim=in_dim_static,
            embedding_dim=1,
            seed=seed,
            nonlin=nonlin,
        )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim_static + embedding_dim_temporal, fc_out_dim),
            nn.BatchNorm1d(fc_out_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_out_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_temporal: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the multimodal GNN.

        Parameters
        ----------
        x_temporal : torch.Tensor
            Input feature matrix for temporal data of shape (num_nodes, in_dim_temporal).
        x_static : torch.Tensor
            Input feature matrix for static data of shape (num_nodes, in_dim_static).

        Returns
        -------
        torch.Tensor
            Output prediction of shape (num_nodes, 1).
        """
        temporal_embedding = self.temporal_branch(x_temporal).squeeze()
        static_embedding = self.static_branch(x_static).squeeze()
        # print(temporal_embedding.shape)
        # print(static_embedding.shape)
        combined_embedding = torch.cat([temporal_embedding, static_embedding], dim=1)
        # print(combined_embedding.shape)
        output = self.classifier(combined_embedding)
        return output
