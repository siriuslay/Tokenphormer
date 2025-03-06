# ----------cora-ARGS---------------------
dataset = "cora"
num_walks = 1
num_class = 7
diameter = 19
node_num = 2708

pretrain_train_path = './TOKENIZED_GRAPH_DATASET/'+dataset+'/pretrain_train.txt'
pretrain_dev_path = './TOKENIZED_GRAPH_DATASET/'+dataset+'/pretrain_dev.txt'

max_seq_length = 256
do_train = True
do_lower_case = True
train_batch_size = 64
eval_batch_size = 64
learning_rate = 1e-4
num_train_epochs = 6 # 6
warmup_proportion = 0.1
no_cuda = False
local_rank = -1   # 非分布式模式
seed = 42
gradient_accumulation_steps = 1
fp16 = False
loss_scale = 0.
bert_config_json = './TOKENIZED_GRAPH_DATASET/'+dataset+ "/bert_config.json"
vocab_file = './TOKENIZED_GRAPH_DATASET/'+dataset+ "/vocab.txt"
output_dir = "./PRETRAINED_MODELS/" + dataset + '/'
masked_lm_prob = 0.15
max_predictions_per_seq = 20


# ----------pubmed-ARGS---------------------
# dataset = "pubmed"
# num_walks = 100
# num_class = 7
# diameter = 18
# node_num = 19716

# pretrain_train_path = './TOKENIZED_GRAPH_DATASET/'+dataset+'/pretrain_train.txt'
# pretrain_dev_path = './TOKENIZED_GRAPH_DATASET/'+dataset+'/pretrain_dev.txt'

# max_seq_length = 256
# do_train = True
# do_lower_case = True
# train_batch_size = 64
# eval_batch_size = 64
# learning_rate = 1e-4
# num_train_epochs = 6 # 6
# warmup_proportion = 0.1
# no_cuda = False
# local_rank = -1
# seed = 42
# gradient_accumulation_steps = 1
# fp16 = False
# loss_scale = 0.
# bert_config_json = './TOKENIZED_GRAPH_DATASET/'+dataset+ "/bert_config.json"
# vocab_file = './TOKENIZED_GRAPH_DATASET/'+dataset+ "/vocab.txt"
# output_dir = "./PRETRAINED_MODELS/" + dataset + '/'
# masked_lm_prob = 0.15
# max_predictions_per_seq = 20


# ----------flickr-ARGS---------------------
# dataset = "flickr"
# num_walks = 20
# diameter = 4

# pretrain_train_path = './TOKENIZED_GRAPH_DATASET/'+dataset+'/pretrain_train.txt'
# pretrain_dev_path = './TOKENIZED_GRAPH_DATASET/'+dataset+'/pretrain_dev.txt'

# max_seq_length = 256
# do_train = True
# do_lower_case = True
# train_batch_size = 64
# eval_batch_size = 64
# learning_rate = 1e-4
# num_train_epochs = 6 # 6
# warmup_proportion = 0.1
# no_cuda = False
# local_rank = -1
# seed = 42
# gradient_accumulation_steps = 1
# fp16 = False
# loss_scale = 0.
# bert_config_json = './TOKENIZED_GRAPH_DATASET/'+dataset+ "/bert_config.json"
# vocab_file = './TOKENIZED_GRAPH_DATASET/'+dataset+ "/vocab.txt"
# output_dir = "./PRETRAINED_MODELS/" + dataset + '/'
# masked_lm_prob = 0.15
# max_predictions_per_seq = 20

# ----------citeseer-ARGS---------------------
# dataset = "citeseer"
# num_walks = 20
# diameter = 28

# pretrain_train_path = './TOKENIZED_GRAPH_DATASET/'+dataset+'/pretrain_train.txt'
# pretrain_dev_path = './TOKENIZED_GRAPH_DATASET/'+dataset+'/pretrain_dev.txt'

# max_seq_length = 256
# do_train = True
# do_lower_case = True
# train_batch_size = 64
# eval_batch_size = 64
# learning_rate = 1e-4
# num_train_epochs = 3
# warmup_proportion = 0.1
# no_cuda = False
# local_rank = -1
# seed = 42
# gradient_accumulation_steps = 1
# fp16 = False
# loss_scale = 0.
# bert_config_json = "TOKENIZED_GRAPH_DATASET/" + dataset + "/bert_config.json"
# vocab_file = "TOKENIZED_GRAPH_DATASET/" + dataset + "/vocab.txt"
# output_dir = "./PRETRAINED_MODELS/" + dataset + '/'
# masked_lm_prob = 0.15
# max_predictions_per_seq = 20

# ----------photo-ARGS---------------------
# dataset = "photo"
# num_walks = 100
# diameter = 11

# pretrain_train_path = './TOKENIZED_GRAPH_DATASET/'+dataset+'/pretrain_train.txt'
# pretrain_dev_path = './TOKENIZED_GRAPH_DATASET/'+dataset+'/pretrain_dev.txt'

# max_seq_length = 256
# do_train = True
# do_lower_case = True
# train_batch_size = 64
# eval_batch_size = 64
# learning_rate = 1e-4
# num_train_epochs = 3
# warmup_proportion = 0.1
# no_cuda = False
# local_rank = -1
# seed = 42
# gradient_accumulation_steps = 1
# fp16 = False
# loss_scale = 0.
# bert_config_json = "TOKENIZED_GRAPH_DATASET/" + dataset + "/bert_config.json"
# vocab_file = "TOKENIZED_GRAPH_DATASET/" + dataset + "/vocab.txt"
# output_dir = "./PRETRAINED_MODELS/" + dataset + '/'
# masked_lm_prob = 0.15
# max_predictions_per_seq = 20
