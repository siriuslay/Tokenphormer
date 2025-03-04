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
TRAIN_PATH      = DIR_PATH + 'pretrain_train.txt'
DEV_PATH        = DIR_PATH + 'pretrain_dev.txt'
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
    node_to_idx = {}
    cur_node = start_node
    index = 0

    node_to_idx[cur_node] = index
    path.append(index)
    index += 1

    for _ in range(length):
        neighbors = G.successors(cur_node).tolist()
        if not neighbors:
            break  # Stop if no outgoing edges

        target_node = random.choice(neighbors)

        if target_node not in node_to_idx:
            node_to_idx[target_node] = index
            index += 1

        path.append(node_to_idx[target_node])

        if target_node == start_node:
            break  # Stop if it returns to the starting node

        cur_node = target_node

    return path


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
            lengths = np.random.normal(mean, std_dev, numPathsOfNode)
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
                    walk = anonymous_walk(G=G, start_node=i, length=length)
                    walk = [str(x) for x in walk]
                else:
                    print("NO SUCH WALK TYPE")
                
                # save files
                f.write(" ".join(walk) + '\n')


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
    generate_random_walk(G, mean_length, std_dev, isDegree=False, randomWalkType="non-backtrackingWithLength", num_walks=100, filePath=TRAIN_PATH, maxLength=128)
    generate_random_walk(G, mean_length, std_dev, isDegree=False, randomWalkType="non-backtrackingWithLength", num_walks=20, filePath=DEV_PATH, maxLength=128)
    print("done")