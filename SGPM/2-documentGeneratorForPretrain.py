import torch
import os
import dgl
import json
import networkx as nx
import numpy as np
from GraphDataset.data import get_dataset
import pretraining_args as args
import random

# path setting
dataset = args.dataset
DIR_PATH        = 'SGPM/TOKENIZED_GRAPH_DATASET/'+dataset+'/'
TRAIN_PATH      = DIR_PATH + 'pretrain_train_MAW_nodegree.txt'
DEV_PATH        = DIR_PATH + 'pretrain_dev_MAW_nodegree.txt'   # _new_lesswalk
VOCAB_PATH      = DIR_PATH + 'vocab.txt'
CONFIG_PATH     = DIR_PATH + 'bert_config.json'

if not os.path.exists(DIR_PATH):        os.makedirs(DIR_PATH)


def anonymous_walk(G, start_node, length):
    """
    G: DGL graph
    start_node: [int] - start node of the path
    length: [int] - length of the path
    
    Return: a list of int-value nodes --> path 
    
    The path is represented by the order of first appearance of nodes.
    """
    path = []
    path_degree = []
    node_to_idx = {}
    visited_pairs = set()  # 存储已访问的节点对
    cur_node = start_node
    index = 0
    steps_taken = 0  # 计算步数，即使撤回也不会减少

    # start_degree = G.out_degrees(start_node)

    node_to_idx[cur_node] = index
    path.append(index)
    index += 1

    # for _ in range(length):
    while steps_taken < length:
        neighbors = G.successors(cur_node).tolist()
        if not neighbors:
            break  # Stop if no outgoing edges

        target_node = random.choice(neighbors)
        pair = (cur_node, target_node)

        if pair in visited_pairs:
            # 如果 (cur_node, target_node) 之前出现过，则撤销 cur_node
            if len(path) > 1:
                removed_node = path.pop()
                # path.pop()
                if len(path) > 1:
                    # prev_prev_node = path[-2]
                    prev_node = path[-1]
                    idx_to_node = {v:k for k,v in node_to_idx.items()}
                    visited_pairs.discard((idx_to_node[prev_node], idx_to_node[removed_node]))
                    target_node = idx_to_node[prev_node]
        else:
            visited_pairs.add(pair)  # 记录新节点对
            if target_node not in node_to_idx:
                node_to_idx[target_node] = index
                index += 1

            path.append(node_to_idx[target_node])
            # steps_taken += 1  # 即使撤回，步数仍增加

        # if target_node == start_node:
        #     break  # Stop if it returns to the starting node

        cur_node = target_node
        steps_taken += 1  # 即使撤回，步数仍增加

    idx_to_node = {v:k for k,v in node_to_idx.items()}
    for node in path:
        idxnode = idx_to_node[node]
        deg = G.out_degrees(idxnode)
        path_degree.append(deg)
    # path_degree.append(G.out_degrees(idx_to_node[node]) for node in path)
    return path, path_degree


def memorized_anonymous_walk(G, start_node, length, num_walks):
    """
    G: DGL graph
    start_node: [int] - start node of the path
    length: [int] - length of the path
    
    Return: a list of int-value nodes --> path 
    
    The paths are represented by the order of first appearance of nodes in all paths.
    """
    node_to_idx = {}  # 所有路径共享的节点索引映射
    all_paths = []     # 存储所有路径的匿名序列
    all_degrees = []   # 存储所有路径的节点度序列

    # 确保起始节点首先被分配索引
    if start_node not in node_to_idx:
        node_to_idx[start_node] = len(node_to_idx)

    for _ in range(num_walks):
        path = []                  # 当前路径的匿名序列
        concrete_path = []         # 当前路径的实际节点
        visited_pairs = set()      # 当前路径的已访问边集合
        steps_taken = 0            # 当前路径已走步数
        
        # 初始化当前节点
        cur_node = start_node
        path.append(node_to_idx[cur_node])
        concrete_path.append(cur_node)

        while steps_taken < length:
            # 获取当前节点的邻居
            neighbors = G.successors(cur_node).tolist()
            
            # 如果没有出边则终止
            if not neighbors:
                break

            # 随机选择下一个节点
            target_node = random.choice(neighbors)
            edge_pair = (cur_node, target_node)

            # 处理边重复的情况
            if edge_pair in visited_pairs:
                # 执行回溯：移除当前节点
                if len(path) > 1:
                    removed_idx = path.pop()
                    removed_node = concrete_path.pop()
                    
                    # 移除对应的边记录
                    prev_node = concrete_path[-1] if concrete_path else None
                    visited_pairs.discard((prev_node, removed_node))
                    
                    # 回退到上一个节点
                    cur_node = prev_node
            else:
                # 添加新节点到索引映射
                if target_node not in node_to_idx:
                    node_to_idx[target_node] = len(node_to_idx)
                
                # 更新路径和边记录
                path.append(node_to_idx[target_node])
                concrete_path.append(target_node)
                visited_pairs.add(edge_pair)
                cur_node = target_node

            steps_taken += 1

        # 生成当前路径的度序列
        idx_to_node = {v: k for k, v in node_to_idx.items()}
        path_degree = [G.out_degrees(idx_to_node[idx]) for idx in path]
        
        # 保存结果
        all_paths.append(path)
        all_degrees.append(path_degree)

    return all_paths, all_degrees


def generate_anonymous_walk(G, mean, std_dev, isDegree=False, randomWalkType="non-tracking", num_walks=20, filePath=TRAIN_PATH, maxLength=250):
    """
    Function to generate a random walk with a length following a normal distribution
    
    G: DGL graph
    mean & std_dev: parameters for normal distribution
    isDegree: [boolean] should the number of paths in direct proportion to degree?
    randomWalkType: [String - "uniform", "non-tracking"] type of random walk
    num_walks: if degree == False, a node generate num_walks paths, else num_walks*degree paths
    filePath: the path to save files
    """     
    print("---> generate setting <---")
    if isDegree == True:    print("path number: degree num")
    else:                   print("path number: fixed num")
    print("walk type:   " + randomWalkType)
            
    # generate random walk paths
    with open(filePath, "w") as f:
        for i in range(G.nodes().size(dim=0)):
            # set path number and length distribution
            numPathsOfNode = num_walks
            if isDegree == True:
                degree = len(G.successors(i)) if len(G.successors(i))<100 else 100
                numPathsOfNode = degree * num_walks

            # generate memorized anonymous path
            f.write(str(i) + '\n')
            walks, degrees = memorized_anonymous_walk(G=G, start_node=i, length=mean, num_walks=numPathsOfNode)
            for idx, walk in enumerate(walks):
                walk = [str(x) for x in walk]
                degree = [str(d) for d in degrees[idx]]

                # save files
                f.write(" ".join(walk) + '\n')
                # f.write(" ".join(walk) + '   ||   ' + " ".join(degree) + '\n')

            # lengths = np.random.normal(mean, std_dev, numPathsOfNode)   # 生成num_walk个长度
            # lengths = [int(x) for x in list(lengths)]
            
            # # generate random path
            # for length in lengths:
                
            #     # random walk
            #     if randomWalkType == "non-tracking":
            #         walk = non_tracking_random_walk(G=G, start_node=i, length=length)
            #         walk = [str(x) for x in walk]
            #     elif randomWalkType == "uniform":
            #         walk = dgl.sampling.random_walk(G, i, length=length)
            #         walk = [str(x) for x in walk[0].tolist()[0]] 
            #     elif randomWalkType == "non-backtracking":
            #         walk = non_backtracking_random_walk(G=G, start_node=i, maxLength=maxLength)
            #         walk = [str(x) for x in walk]
            #     elif randomWalkType == "non-backtrackingWithLength":
            #         walk = non_backtracking_random_walk_with_length(G=G, start_node=i, length=length)
            #         walk = [str(x) for x in walk]
            #     elif randomWalkType == "anonymous":
            #         walk = anonymous_walk(G=G, start_node=i, length=length)
            #         walk = [str(x) for x in walk]
            #     else:
            #         print("NO SUCH WALK TYPE")
                
                # save files
                # f.write(" ".join(walk) + '\n')

def non_backtracking_random_walk(G, start_node, maxLength):
    """
    G: DGL graph
    start_node: [int] - start node of the path
    maxLength: [int] - max length of the path
    
    Return: a list of int-value nodes --> path 
    
    the path would not go back immediately
    """
    path = []
    pre_node = None
    cur_node = start_node
    path.append(int(start_node))
    
    neighborsFirst = G.successors(start_node)
    if len(neighborsFirst)==0 or int(neighborsFirst[0])==-1:
        return path
    
    for i in range(maxLength):
        neighbors = G.successors(cur_node).tolist()
        lenNei = len(neighbors)

        if lenNei == 1:
            target_node = neighbors[0]
        else:
            if pre_node != None:
                neighbors.remove(pre_node)
            target_node = random.choice(neighbors)
        
        path.append(int(target_node))
        if(target_node == start_node):
            break
        pre_node = cur_node
        cur_node = target_node
    
    # print(path)
    return path

def non_backtracking_random_walk_with_length(G, start_node, length):
    """
    G: DGL graph
    start_node: [int] - start node of the path
    length: [int] - length of the path
    
    Return: a list of int-value nodes --> path 
    
    the path would not go back immediately
    """
    path = []
    pre_node = None
    cur_node = start_node
    path.append(int(start_node))
    
    neighborsFirst = G.successors(start_node)
    if len(neighborsFirst)==0 or int(neighborsFirst[0])==-1:
        return path
    
    for i in range(length):
        neighbors = G.successors(cur_node).tolist()
        lenNei = len(neighbors)

        if lenNei == 1:
            target_node = neighbors[0]
        else:
            if pre_node != None:
                neighbors.remove(pre_node)
            target_node = random.choice(neighbors)
        
        path.append(int(target_node))
        pre_node = cur_node
        cur_node = target_node
    
    return path

def non_tracking_random_walk(G, start_node, length):
    """
    G: DGL graph
    start_node: [int] - start node of the path
    length: [int] - length of the path
    
    Return: a list of int-value nodes --> path 
    
    In ideal situation, the path would not have repeated node
    In special situation (all neighbors of current node are in the path), randomly choose one neighbor as next node in the path
    """
    path = []
    pre_node = [start_node]
    cur_node = start_node
    path.append(start_node)
    
    if len(G.successors(start_node)) == 0:
        return path
    
    for i in range(length):
        neighbors = G.successors(cur_node)
        
        neighbors2 = []
        for neighbor in neighbors:
            if neighbor not in pre_node:
                neighbors2.append(neighbor)

        if len(neighbors2) > 0:
            target_node = random.choice(neighbors2)
        else:
            target_node = random.choice(neighbors)
        
        path.append(int(target_node))
        pre_node.append(target_node)
        cur_node = target_node
        
    return path

def generate_random_walk(G, mean, std_dev, isDegree=False, randomWalkType="non-tracking", num_walks=20, filePath=TRAIN_PATH, maxLength=250):
    """
    Function to generate a random walk with a length following a normal distribution
    
    G: DGL graph
    mean & std_dev: parameters for normal distribution
    isDegree: [boolean] should the number of paths in direct proportion to degree?
    randomWalkType: [String - "uniform", "non-tracking"] type of random walk
    num_walks: if degree == False, a node generate num_walks paths, else num_walks*degree paths
    filePath: the path to save files
    """     
    print("---> generate setting <---")
    if isDegree == True:    print("path number: degree num")
    else:                   print("path number: fixed num")
    print("walk type:   " + randomWalkType)
            
    # generate random walk paths
    with open(filePath, "w") as f:
        for i in range(G.nodes().size(dim=0)):
            # set path number and length distribution
            numPathsOfNode = num_walks
            if isDegree == True:
                degree = len(G.successors(i)) if len(G.successors(i))<100 else 100
                numPathsOfNode = degree * num_walks
            lengths = np.random.normal(mean, std_dev, numPathsOfNode)   # 生成num_walk个长度
            lengths = [int(x) for x in list(lengths)]
            
            # generate path
            for length in lengths:
                
                # random walk
                if randomWalkType == "non-tracking":
                    walk = non_tracking_random_walk(G=G, start_node=i, length=length)
                    walk = [str(x) for x in walk]
                elif randomWalkType == "uniform":
                    walk = dgl.sampling.random_walk(G, i, length=length)
                    walk = [str(x) for x in walk[0].tolist()[0]] 
                elif randomWalkType == "non-backtracking":
                    walk = non_backtracking_random_walk(G=G, start_node=i, maxLength=maxLength)
                    walk = [str(x) for x in walk]
                elif randomWalkType == "non-backtrackingWithLength":
                    walk = non_backtracking_random_walk_with_length(G=G, start_node=i, length=length)
                    walk = [str(x) for x in walk]
                elif randomWalkType == "anonymous":
                    walk, degree = anonymous_walk(G=G, start_node=i, length=length)
                    walk = [str(x) for x in walk]
                    degree = [str(d) for d in degree]
                else:
                    print("NO SUCH WALK TYPE")
                
                # save files
                f.write(" ".join(walk) + '\n')
                # f.write(" ".join(walk) + '   ||   ' + " ".join(degree) + '\n')


if __name__ == "__main__":
    dataset = args.dataset
    # G, features, nclass, train, val, test, labels = preprocess_data(dataset, 60)
    # G, adj, features, labels, idx_train, idx_val, idx_test = get_dataset(dataset)
    G, features, labels, idx_train, idx_val, idx_test = get_dataset(args.dataset, 60)

    t_num = args.num_walks
    diameter = args.diameter
    # mean_length = diameter // 2
    
    # # compute the diameter of the graph
    # nx_g = dgl.to_networkx(G)
    # try:
    #     diameter = nx.diameter(nx_g.to_undirected())
    # except:
    #     print("graph is not connected, please input the diameter in hands.")
    # finally:
    #     diameter = input("Graph Diameter of " + dataset + ":")


    mean_length = diameter // 2
    std_dev = 1 # Adjust the divisor as per desired spread

    print("*" * 20 + 'generating random walk' + "*" * 20)
    # generate_random_walk(G, mean_length, std_dev, isDegree=False, randomWalkType="non-backtrackingWithLength", num_walks=10, filePath=TRAIN_PATH, maxLength=128)
    # generate_random_walk(G, mean_length, std_dev, isDegree=False, randomWalkType="non-backtrackingWithLength", num_walks=2, filePath=DEV_PATH, maxLength=128)
    generate_anonymous_walk(G, mean_length, std_dev, isDegree=False, randomWalkType="anonymous", num_walks=10, filePath=TRAIN_PATH, maxLength=128)
    generate_anonymous_walk(G, mean_length, std_dev, isDegree=False, randomWalkType="anonymous", num_walks=2, filePath=DEV_PATH, maxLength=128)
    # generate_random_walk(G, mean_length, std_dev, isDegree=False, randomWalkType="anonymous", num_walks=10, filePath=TRAIN_PATH, maxLength=128)
    # generate_random_walk(G, mean_length, std_dev, isDegree=False, randomWalkType="anonymous", num_walks=2, filePath=DEV_PATH, maxLength=128)
    print("done")

    