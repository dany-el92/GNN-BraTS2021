import numpy as np
import os
from dotenv import load_dotenv
from tqdm import tqdm
import datetime
import warnings
import random
import pickle
import itertools
import dgl
import torch
import pandas as pd
import argparse
from utilities import predict, save_settings
from compute_metrics import calculate_node_dices, voxel_wise_batch_score


# Create the parser 
parser = argparse.ArgumentParser() 
# Add an argument for the model 
parser.add_argument('--model', type=str, required=True, help="Name of the model") 
# Parse the arguments 
args = parser.parse_args() 
# You can now access the model argument with args.model 
print(f'Training with model: {args.model}')


load_dotenv()
dataset_pickle_path = os.getenv('DATASET_PICKLE_PATH')
val_model_path = os.getenv('MODEL_PATH')
metrics_path = os.getenv('METRICS_TESTING_SAVE_PATH')

timestamp = datetime.datetime.now()

# Ignore UserWarning related to TypedStorage deprecation
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, message="To copy construct from a tensor")
os.environ["DGLBACKEND"] = "pytorch"


####### LOAD THE DATASET  AND SPLIT TRAIN - TEST - VAL ########


with open(dataset_pickle_path, 'rb') as f:
    dataset = pickle.load(f)
random.seed(42)  # Set the random seed to ensure reproducibility

# Shuffle the dataset
random.shuffle(dataset)

# Calculate the indices for splitting
train_split = int(len(dataset) * 0.7)
val_split = int(len(dataset) * 0.9)  # This is 90% because we're taking 20% of the remaining data after the training split

# Split the data
train_data = dataset[:train_split]
val_data = dataset[train_split:val_split]
test_data = dataset[val_split:]  # The remaining 10% of the data

print(f"Train data size: {len(train_data)}")
print(f"Validation data size: {len(val_data)}")
print(f"Test data size: {len(test_data)}")


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)

# Create your batches
train_batches = list(grouper(train_data, 6))
val_batches = list(grouper(val_data, 6))
test_batches = list(grouper(test_data, 6))


def testing_batch(timestamp, dgl_test_graphs, model):
    
    print(f'Testing started at: {timestamp}')

    metrics = []

    for batch in tqdm(dgl_test_graphs):
        bg = dgl.batch([data[0] for data in batch])  # batched graph
        features = torch.cat([torch.tensor(data[1]).float() for data in batch], dim=0)  # concatenate features
        labels = torch.cat([torch.tensor(data[2]).long() - 1 for data in batch], dim=0)  # Offset the labels and concatenate        
        ids = [data[3] for data in batch]

        pred = predict(bg, features, model)
        labels = labels + 1 

        node_dice_wt, node_dice_ct, node_dice_et = calculate_node_dices(pred, labels)
        # pred[pred == 3] = 0

        mean_scores = voxel_wise_batch_score(pred, ids)
        voxel_dice_wt, voxel_dice_ct, voxel_dice_et, \
            voxel_hd95_wt, voxel_hd95_ct, voxel_hd95_et = mean_scores
    
        metrics.append({
            'dice_score_node_WT': node_dice_wt,
            'dice_score_node_CT': node_dice_ct,
            'dice_score_node_ET': node_dice_et,
            'dice_score_voxel_WT': voxel_dice_wt,
            'dice_score_voxel_CT': voxel_dice_ct,
            'dice_score_voxel_ET': voxel_dice_et,
            'hd95_voxel_WT': voxel_hd95_wt,
            'hd95_voxel_CT': voxel_hd95_ct,
            'hd95_voxel_ET': voxel_hd95_et,
        })

        print(f"dice_score_node_WT: {node_dice_wt} | dice_score_node_CT: {node_dice_ct} | dice_score_node_ET: {node_dice_et} | \
                dice_score_voxel_WT: {voxel_dice_wt} | dice_score_voxel_CT: {voxel_dice_ct} | dice_score_voxel_ET: {voxel_dice_et} | hd95_voxel_WT: {voxel_hd95_wt} | \
                hd95_voxel_CT: {voxel_hd95_ct} | hd95_voxel_ET: {voxel_hd95_et}")
        
        # Save metrics to a CSV file
        df_metrics = pd.DataFrame(metrics)
        string_timestamp = timestamp.strftime("%Y%m%d-%H%M%S")
        df_metrics.to_csv(f'{metrics_path}/{string_timestamp}/testing_metrics_{string_timestamp}.csv', index=False)

            


#########################################
from GATSage import GraphSage, GAT, GIN, ChebNet

in_feats = 20
layer_sizes = [512, 512, 512]
n_classes = 4
heads = [6, 6, 6, 6, 6, 6]
residuals = [True, True, True, True, True, True]

patience = 10 # number of epochs to wait for improvement before stopping
lr = 0.0005
weight_decay = 0.0001
gamma = 0.98

val_dropout = 0.2
val_feat_drop = 0
val_attn_drop = 0

val_k = 3

# Create model
if args.model == 'GraphSage':
    test_model = GraphSage(in_feats, layer_sizes, n_classes, aggregator_type = 'pool', dropout = val_dropout)
elif args.model == 'GAT':
    test_modelmodel = GAT(in_feats, layer_sizes, n_classes, heads, residuals, feat_drop = val_feat_drop, attn_drop = val_attn_drop)
elif args.model == 'GIN':
    test_model = GIN(in_feats, layer_sizes, n_classes, dropout = val_dropout)
elif args.model == 'Cheb':
    test_model = ChebNet(in_feats, layer_sizes, n_classes, k = val_k, dropout = val_dropout)

save_settings(timestamp, test_model, patience, lr, weight_decay, gamma, args.model, heads, residuals, \
              val_dropout, layer_sizes, in_feats, n_classes, val_feat_drop, val_attn_drop, dataset_pickle_path, val_k, model_path = val_model_path)


test_model.load_state_dict(torch.load(val_model_path))

testing_batch(timestamp, test_batches, test_model)

