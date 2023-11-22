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
DIR_PATH        = './TOKENIZED_GRAPH_DATASET/'+dataset+'/'
TRAIN_PATH      = DIR_PATH + 'pretrain_train.txt'
DEV_PATH        = DIR_PATH + 'pretrain_dev.txt'
VOCAB_PATH      = DIR_PATH + 'vocab.txt'
CONFIG_PATH     = DIR_PATH + 'bert_config.json'

if not os.path.exists(DIR_PATH):        os.makedirs(DIR_PATH)

def generate_vocab(nodeNum, dataset, filePath=VOCAB_PATH):
    sign = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    with open(filePath, "w") as f:
        for s in sign:
            f.write(s + "\n")
        for i in range(nodeNum):
            f.write(str(i) + "\n")

def generate_config(nodeNum, filePath):
    dict = {
        "attention_probs_dropout_prob": 0.1, 
        "directionality": "bidi", 
        "hidden_act": "gelu", 
        "hidden_dropout_prob": 0.1, 
        "hidden_size": 768, 
        "initializer_range": 0.02, 
        "intermediate_size": 3072, 
        "max_position_embeddings": 512, 
        "num_attention_heads": 12, 
        "num_hidden_layers": 4, 
        "pooler_fc_size": 768, 
        "pooler_num_attention_heads": 12, 
        "pooler_num_fc_layers": 3, 
        "pooler_size_per_head": 128, 
        "pooler_type": "first_token_transform", 
        "type_vocab_size": 2, 
        "vocab_size": nodeNum+5
    }
    with open(filePath, "w") as f:
        json.dump(dict, f)


if __name__ == "__main__":
    dataset = args.dataset
    # G, features, nclass, train, val, test, labels = preprocess_data(dataset, 60)
    # G, adj, features, labels, idx_train, idx_val, idx_test = get_dataset(dataset)
    G, features, labels, idx_train, idx_val, idx_test = get_dataset(args.dataset, 60)

    nodes_num = G.nodes().shape[0]

    print("*"*20 + 'generating bert config'+ "*"*20)
    generate_config(nodes_num, CONFIG_PATH)
    print('done')

    print("*"*20 + 'generating vocab'+ "*"*20)
    generate_vocab(nodes_num, dataset, VOCAB_PATH)
    print('done')

    
    
    
            

            
            
            
    
            
            

            
            
            
    
