import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import os
import networkx as nx
import metis
import numpy as np
import scipy.sparse as sp
import torch
import random
from dgl import DGLGraph, RandomWalkPE
import dgl

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def torch_sparse_tensor_to_sparse_mx(torch_sparse):
    """Convert a torch sparse tensor to a scipy sparse matrix."""

    m_index = torch_sparse._indices().numpy()
    row = m_index[0]
    col = m_index[1]
    data = torch_sparse._values().numpy()

    sp_matrix = sp.coo_matrix((data, (row, col)), shape=(torch_sparse.size()[0], torch_sparse.size()[1]))

    return sp_matrix

def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """
    # Laplacian
    #adjacency_matrix(transpose, scipy_fmt="csr")
    # A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    A = g.adj_external(scipy_fmt="csr").astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with scipy
    #EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2) # for 40 PEs
    EigVec = EigVec[:, EigVal.argsort()] # increasing order
    lap_pos_enc = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 

    return lap_pos_enc


def metis_partition_dglgraph(g, n_patches=50):
    num_nodes = g.num_nodes()

    if num_nodes < n_patches:
        membership = torch.randperm(n_patches)[:num_nodes]
    else:
        # 提取 DGL 图的边信息
        src, dst = g.edges()
        edge_list = list(zip(src.tolist(), dst.tolist()))

        # 转换为 NetworkX 图
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(edge_list)

        # METIS 分区
        cuts, membership = metis.part_graph(G, n_patches, recursive=True)

    assert len(membership) >= num_nodes
    membership = torch.tensor(membership[:num_nodes])

    # 组织 Patch
    patch = []
    max_patch_size = -1
    for i in range(n_patches):
        patch.append(torch.where(membership == i)[0].tolist())
        max_patch_size = max(max_patch_size, len(patch[-1]))

    # 统一 Patch 大小
    for i in range(len(patch)):
        if len(patch[i]) < max_patch_size:
            patch[i] += [num_nodes] * (max_patch_size - len(patch[i]))

    patch = torch.tensor(patch)
    return patch