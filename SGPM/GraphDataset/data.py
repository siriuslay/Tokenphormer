import utils
import dgl
import torch
from ogb.nodeproppred import DglNodePropPredDataset
from ogb.linkproppred import DglLinkPropPredDataset
import scipy.sparse as sp
import os.path
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import  CoraFullDataset, AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset,CoauthorCSDataset,CoauthorPhysicsDataset
import random
import pickle as pkl
import numpy as np
from GraphDataset.make_dataset import get_train_val_test_split
from sklearn.preprocessing import StandardScaler
from GraphDataset.proj.data.preprocess import preprocess_data
import GraphDataset.proj.functions as uf
from sklearn.model_selection import train_test_split
# from cache_sample import cache_sample_rand_csr

def get_dataset(dataset, split_seed=0):
    if dataset in {"arxiv", "products", "proteins", "papers100M", "mag"}:
        if dataset == "arxiv":
            dataset = DglNodePropPredDataset(name="ogbn-arxiv")
        elif dataset == "products":
            dataset = DglNodePropPredDataset(name="ogbn-products")
        elif dataset == "proteins":
            dataset = DglNodePropPredDataset(name="ogbn-proteins")
        elif dataset == "papers100M":
            dataset = DglNodePropPredDataset(name="ogbn-papers100M")
        elif dataset == "mag":
            dataset = DglNodePropPredDataset(name="ogbn-mag")
        split_idx = dataset.get_idx_split()
        graph, labels = dataset[0]
        features = graph.ndata['feat']
        adj = graph.adj(scipy_fmt="csr")
        # adj = cache_sample_rand_csr(adj, s_len)
        # print(labels)

        idx_train = split_idx['train']
        idx_val = split_idx['valid']
        idx_test = split_idx['test']

        graph = dgl.from_scipy(adj)

        adj = utils.sparse_mx_to_torch_sparse_tensor(adj)

        labels = labels.reshape(-1)

        # RWPE
        # lpe = utils.randomwalk_positional_encoding(adj, rw_dim)
        # features = torch.cat((features, lpe.ndata['PE']), dim=1)

        # LPE
        lpe = utils.laplacian_positional_encoding(graph, 3) 
        features = torch.cat((features, lpe), dim=1)

    elif dataset in {"pubmed", "corafull", "computer", "photo", "cs", "physics","cora", "citeseer"}:
        file_path = "SGPM/GraphDataset/dataset/"+dataset+".pt"

        data_list = torch.load(file_path)

        # data_list = [adj, features, labels, idx_train, idx_val, idx_test]
        adj = data_list[0]
        features = data_list[1]
        labels = data_list[2]

        idx_train = data_list[3]
        idx_val = data_list[4]
        idx_test = data_list[5]

        if dataset == "pubmed":
            graph = PubmedGraphDataset()[0]
        elif dataset == "corafull":
            graph = CoraFullDataset()[0]
        elif dataset == "computer":
            graph = AmazonCoBuyComputerDataset()[0]
        elif dataset == "photo":
            graph = AmazonCoBuyPhotoDataset()[0]
        elif dataset == "cs":
            graph = CoauthorCSDataset()[0]
        elif dataset == "physics":
            graph = CoauthorPhysicsDataset()[0]
        elif dataset == "cora":
            graph = CoraGraphDataset()[0]
        elif dataset == "citeseer":
            graph = CiteseerGraphDataset()[0]

        graph = dgl.to_bidirected(graph)

        # RWPE
        # lpe = utils.randomwalk_positional_encoding(adj, rw_dim)
        # features = torch.cat((features, lpe.ndata['PE']), dim=1)

        # # LPE
        lpe = utils.laplacian_positional_encoding(graph, 3) 
        features = torch.cat((features, lpe), dim=1)

        # col_normalize
        # features = col_normalize(features)
        # features = torch.tensor(features)

        
    # elif dataset == 'aminer':
    #     path = "GraphDataset/dataset/"+dataset+"/"
    #     adj = pkl.load(open(os.path.join(path, "{}.adj.sp.pkl".format(dataset)), "rb"))
    #     features = pkl.load(
    #         open(os.path.join(path, "{}.features.pkl".format(dataset)), "rb"))
    #     labels = pkl.load(
    #         open(os.path.join(path, "{}.labels.pkl".format(dataset)), "rb"))
    #     random_state = np.random.RandomState(split_seed)
    #     idx_train, idx_val, idx_test = get_train_val_test_split(
    #         random_state, labels, train_examples_per_class=20, val_examples_per_class=30)
    #     # idx_unlabel = np.concatenate((idx_val, idx_test))
    #     features = col_normalize(features)
        
    #     labels = torch.tensor(labels)
    #     idx_train = torch.tensor(idx_train)
    #     idx_val = torch.tensor(idx_val)
    #     idx_test = torch.tensor(idx_test)
        
        
    #     graph = dgl.from_scipy(adj)
    #     # print(graph)
    #     graph = dgl.to_bidirected(graph)
        
    #     # lpe = utils.laplacian_positional_encoding(graph, pe_dim) 
       
    #     # features = torch.cat((features, lpe), dim=1)

    #     adj = utils.sparse_mx_to_torch_sparse_tensor(adj)        
    #     labels = torch.argmax(labels, -1)
        
    elif dataset in ['flickr']:
        train_percentage = 60
        load_default_split = train_percentage <= 0
        DATA_PATH = f'GraphDataset/dataset'
        adj_orig = pkl.load(open(f'{DATA_PATH}/{dataset}/{dataset}_adj.pkl', 'rb'))  # sparse
        features = pkl.load(open(f'{DATA_PATH}/{dataset}/{dataset}_features.pkl', 'rb'))  # sparase
        labels = pkl.load(open(f'{DATA_PATH}/{dataset}/{dataset}_labels.pkl', 'rb'))  # tensor
        if torch.is_tensor(labels):
            labels = labels.numpy()

        if load_default_split:
            tvt_nids = pkl.load(open(f'{DATA_PATH}/{dataset}/{dataset}_tvt_nids.pkl', 'rb'))  # 3 array
            train = tvt_nids[0]
            val = tvt_nids[1]
            test = tvt_nids[2]
        else:
            train, val, test = stratified_train_test_split(np.arange(len(labels)), labels, len(labels),
                                                           train_percentage)

        
        adj_orig = adj_orig.tocoo()
        U = adj_orig.row.tolist()
        V = adj_orig.col.tolist()
        g = dgl.graph((U, V))
        g = dgl.to_simple(g)
        g = dgl.remove_self_loop(g)
        graph = dgl.to_bidirected(g)

        if dataset in ['airport']:
            features = row_normalization(features)

        if sp.issparse(features):
            features = torch.FloatTensor(features.toarray())
        else:
            features = torch.FloatTensor(features)

        labels = torch.LongTensor(labels)
        idx_train = torch.LongTensor(train)
        idx_val = torch.LongTensor(val)
        idx_test = torch.LongTensor(test)

    elif dataset in {"aminer", "reddit", "Amazon2M"}:
        file_dir = 'GraphDataset/dataset'
        file_path = file_dir + dataset + '.pt'
        if dataset == 'aminer':

            adj = pkl.load(open(os.path.join(file_dir, "{}.adj.sp.pkl".format(dataset)), "rb"))
            features = pkl.load(
                open(os.path.join(file_dir, "{}.features.pkl".format(dataset)), "rb"))
            labels = pkl.load(
                open(os.path.join(file_dir, "{}.labels.pkl".format(dataset)), "rb"))
            random_state = np.random.RandomState(split_seed)
            idx_train, idx_val, idx_test = get_train_val_test_split(
                random_state, labels, train_examples_per_class=20, val_examples_per_class=30)
            idx_unlabel = np.concatenate((idx_val, idx_test))
            features = col_normalize(features)
        elif dataset in ['reddit']:
            adj = sp.load_npz(os.path.join(file_dir, '{}_adj.npz'.format(dataset)))
            features = np.load(os.path.join(file_dir, '{}_feat.npy'.format(dataset)))
            labels = np.load(os.path.join(file_dir, '{}_labels.npy'.format(dataset))) 
            print(labels.shape, list(np.sum(labels, axis=0)))
            random_state = np.random.RandomState(split_seed)
            idx_train, idx_val, idx_test = get_train_val_test_split(
                random_state, labels, train_examples_per_class=20, val_examples_per_class=30)    
            idx_unlabel = np.concatenate((idx_val, idx_test))
            print(dataset, features.shape)

        elif dataset in ['Amazon2M']:
            adj = sp.load_npz(os.path.join(file_dir, '{}_adj.npz'.format(dataset)))
            features = np.load(os.path.join(file_dir, '{}_feat.npy'.format(dataset)))
            labels = np.load(os.path.join(file_dir, '{}_labels.npy'.format(dataset)))
            print(labels.shape, list(np.sum(labels, axis=0)))
            random_state = np.random.RandomState(split_seed)
            class_num = labels.shape[1]
            idx_train, idx_val, idx_test = get_train_val_test_split(random_state, labels, train_size=20*class_num, val_size=30 * class_num)
            idx_unlabel = np.concatenate((idx_val, idx_test))

        adj = adj + sp.eye(adj.shape[0])
        D1 = np.array(adj.sum(axis=1))**(-0.5)
        D2 = np.array(adj.sum(axis=0))**(-0.5)
        D1 = sp.diags(D1[:, 0], format='csr')
        D2 = sp.diags(D2[0, :], format='csr')

        A = adj.dot(D1)
        A = D2.dot(A)
        adj = A

        features = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor(labels)
        idx_train = torch.tensor(idx_train)
        idx_val = torch.tensor(idx_val)
        idx_test = torch.tensor(idx_test)
        
    elif dataset in ['dblp']:
        fname = f'GraphDataset/dataset/dblp/processed_dblp.pickle'
        if os.path.exists(fname):
            from torch_geometric.datasets import CitationFull
            import torch_geometric.transforms as T
            data = CitationFull(root=f'./dataset', name=dataset, transform=T.NormalizeFeatures())[0]
            edges = data.edge_index
            features = data.x.numpy()
            labels = data.y.numpy()
            data_dict = {'edges': edges, 'features': features, 'labels': labels}
            uf.save_pickle(data_dict, fname)
        else:
            data_dict = uf.load_pickle(fname)
        edges, features, labels = data_dict['edges'], data_dict['features'], data_dict['labels']
        train, val, test = stratified_train_test_split(np.arange(len(labels)), labels, len(labels), 60)

        U = edges[0]
        V = edges[1]
        g = dgl.graph((U, V))
        g = dgl.to_simple(g)
        g = dgl.remove_self_loop(g)
        graph = dgl.to_bidirected(g)

        features = torch.FloatTensor(features)
        labels = torch.LongTensor(labels)
        idx_train = torch.LongTensor(train)
        idx_val = torch.LongTensor(val)
        idx_test = torch.LongTensor(test)
        
        lpe = utils.laplacian_positional_encoding(graph, 3) 
        features = torch.cat((features, lpe), dim=1)
        
    return graph, features, labels, idx_train, idx_val, idx_test


def col_normalize(mx):
    """Column-normalize sparse matrix"""
    scaler = StandardScaler()

    mx = scaler.fit_transform(mx)

    return mx

# def obtain_dataset(dataset, split_seed=0):
#     if dataset in ["cora", "pubmed", "citeseer", "computer", "photo"]:
#         G, adj, features, labels, idx_train, idx_val, idx_test = get_dataset(dataset, split_seed)
#     else:
#         G, features, nclass, idx_train, idx_val, idx_test, labels = preprocess_data(dataset, split_seed)
    
#     return G, features, idx_train, idx_val, idx_test, labels

def stratified_train_test_split(label_idx, labels, n_nodes, train_rate, dataset=''):
    if dataset == 'cora':
        seed = 0
    else:
        seed = 2021
    n_train_nodes = int(train_rate / 100 * n_nodes)
    test_rate_in_labeled_nodes = (len(labels) - n_train_nodes) / len(labels)
    train_idx, test_and_valid_idx = train_test_split(
        label_idx, test_size=test_rate_in_labeled_nodes, random_state=seed, shuffle=True, stratify=labels)
    valid_idx, test_idx = train_test_split(
        test_and_valid_idx, test_size=.5, random_state=seed, shuffle=True, stratify=labels[test_and_valid_idx])
    return train_idx, valid_idx, test_idx

def row_normalization(mat):
    """Row-normalize sparse matrix"""
    row_sum = np.array(mat.sum(1))
    r_inv = np.power(row_sum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mat = r_mat_inv.dot(mat)
    return mat