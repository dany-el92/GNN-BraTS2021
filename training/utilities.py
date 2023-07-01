import numpy as np
import networkx as nx
import nibabel as nib
import pickle
import matplotlib.pyplot as plt
import torch
import dgl
import dgl.data
import plotly.graph_objs as go
import sklearn.metrics as metrics
from sklearn.metrics import precision_score, recall_score, f1_score
import json
import os
from collections import Counter
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()
dataset_path = os.getenv('DATASET_PATH')
metrics_training_path = os.getenv('METRICS_TRAINING_SAVE_PATH')
metrics_testing_path = os.getenv('METRICS_TESTING_SAVE_PATH')


def load_networkx_graph(fp): 
    with open(fp,'r') as f: 
        json_graph = json.loads(f.read()) 
        return nx.readwrite.json_graph.node_link_graph(json_graph) 
    
    
def get_graph(graph_path, id): 
    nx_graph = load_networkx_graph(graph_path) 
    features = np.array([nx_graph.nodes[n]['features'] for n in nx_graph.nodes]) 
    labels = np.array([nx_graph.nodes[n]['label'] for n in nx_graph.nodes]) 
    
    # Mappatura delle etichette 
    label_mapping = {0: 3, 1: 2, 2: 1, 3: 4} 
    labels = np.vectorize(label_mapping.get)(labels) 
    
    G = dgl.from_networkx(nx_graph) 
    n_edges = G.number_of_edges() 
    # normalization 
    degs = G.in_degrees().float() 
    norm = torch.pow(degs, -0.5) 
    norm[torch.isinf(norm)] = 0 
    G.ndata['norm'] = norm.unsqueeze(1) 
    #G.ndata['feat'] = features 
    return (G, features, labels, id)  


def generate_dgl_dataset(dataset_path):
    print(f'generating the dataset form {dataset_path}')
    subdirectories = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    new_graphs = [] 
    for subdir in tqdm(subdirectories):
        subdir_path = os.path.join(dataset_path, subdir)
        try:
            id, flag = get_patient_ids([f"{dataset_path}/{subdir}"])
            for filename in os.listdir(subdir_path):
                file_type = (filename.split("_")[2]).split(".")[0]
                if file_type in ['nxgraph']:
                    quadruple = get_graph(f'{dataset_path}/{subdir}/BraTS2021_{id[0]}_nxgraph.json', id[0]) 
                    new_graphs.append(quadruple)

        except Exception as e:
            print(f'Error: {e}')
    print('Success! The dataset has been generated.')
    with open('full_dataset_with_id.pickle', 'wb') as f:
        pickle.dump(new_graphs, f)
    return new_graphs


def create_batches(data, batch_size):
    # Create batches of graphs, features, labels
    batches = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        graphs, features, labels = zip(*batch)
        
        # convert features and labels to tensors if they are not
        features = [torch.tensor(f) if isinstance(f, np.ndarray) else f for f in features]
        labels = [torch.tensor(l) if isinstance(l, np.ndarray) else l for l in labels]

        batched_graph = dgl.batch(graphs)  # Here we batch graphs
        # Keep features as a list because they have different sizes
        batched_features = features
        batched_labels = labels
        batches.append((batched_graph, batched_features, batched_labels))
    return batches


def batch_dataset(dataset):
    batched_data = []
    current_batch = []
    for item in dataset:
        current_batch.append(item)
        if len(current_batch) == 6:
            batched_data.append(current_batch)
            current_batch = []
    if current_batch:
        # If there are remaining items that don't fill a complete batch
        batched_data.append(current_batch)
    return batched_data


def get_coordinates(tumor_seg, values=[1, 2, 4]):

    coordinates = {}
    
    for value in values:
        # Get the indices where the value is present in the image
        indices = np.where(tumor_seg == value)

        # Convert the indices to a list of tuples representing coordinates
        coords = list(zip(*indices))

        # Add the coordinates to the dictionary
        coordinates[value] = coords

    return coordinates


def get_patient_ids(paths):
    ids = []
    for path in paths:
        splitted_path = path.split("/")
        ids.append(splitted_path[-1].split("_")[1])
    if all(elem == ids[0] for elem in ids):
        return ids, True
    else:
        return ids, False
    

def get_supervoxel_values(slic_image, coordinates_dict):

    supervoxel_values = {}

    for value, coordinates in coordinates_dict.items():
        value_list = []
        for coord in coordinates:
            # Get the value of the supervoxel at the given coordinate
            supervoxel_value = slic_image[coord]

            # Add the value to the list
            value_list.append(supervoxel_value)
        
        # Add the list of supervoxel values to the dictionary
        supervoxel_values[value] = list(np.unique(value_list))

    return supervoxel_values


def add_labels_to_single_graph(tumor_seg, slic_image, graph):

    coords = get_coordinates(tumor_seg)
    labels_supervoxel_dict = get_supervoxel_values(slic_image, coords)

    for label, supervoxel_list in labels_supervoxel_dict.items():
        for supervoxel in supervoxel_list:
            graph.nodes[str(int(supervoxel))]["label"] = label

    for n in graph.nodes():
        try:
            graph.nodes[n]["label"]
        except:
            graph.nodes[n]["label"] = 3
    
    return graph    


def generate_tumor_segmentation_from_graph(predicted_labels, slic): 
    supervoxels = np.unique(slic) 
    label_map = dict(zip(supervoxels, list(predicted_labels)[:len(supervoxels)])) 
    slic = np.vectorize(label_map.get)(slic) 
    return slic
    

def tensor_labels(segmented_image, labels_generated, empty_RAG, id_patient, save=False):
    R = add_labels_to_single_graph(labels_generated, segmented_image, empty_RAG)
    tl = torch.tensor(list(nx.get_node_attributes(R, 'label').values()))
    if save==True:
       torch.save(tl, f'/content/drive/MyDrive/Tesi Progetto/tensor_labels/tensor_label_{id_patient}')
    return tl


def _3Dplotter(numpy_image):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
    ax1.imshow(numpy_image[numpy_image.shape[0] // 2, :, :])
    ax1.set_title("Sagittale")
    ax2.imshow(numpy_image[:, numpy_image.shape[1] // 2, :])
    ax2.set_title("Coronale")
    ax3.imshow(numpy_image[:, :, numpy_image.shape[2] // 2])
    ax3.set_title("Assiale")
    plt.show()


def load_and_assign_tensor_labels(graph_ids, tensor_labels): 
    labeled_graphs = [] 

    for graph_id, tensor_label in zip(graph_ids, tensor_labels): 
        # Load the graph associated with the graph_id 
        file_name = f"/content/drive/MyDrive/Tesi Progetto/graphs/brain_graph_{graph_id}.graphml" 
        graph = nx.read_graphml(file_name) 

        tensor_label = tensor_label.cpu().numpy()
        i = 0
        
        for node in graph.nodes():
            if node == '0':
                graph.nodes[node]['label'] = 0
            else:
                graph.nodes[node]['label'] = tensor_label[i]
            i += 1

        labeled_graphs.append(graph)
    
    j = 0
    predicted_tumor_images = []
    for labeled_graph in labeled_graphs:
        img = nib.load(f"/content/drive/MyDrive/Tesi Progetto/dataset/BraTS2021_{graph_ids[j]}/BraTS2021_{graph_ids[j]}_SLIC.nii.gz")
        slic = img.get_fdata()
        im = generate_tumor_segmentation_from_graph(slic, labeled_graph)
        predicted_tumor_images.append(im)
        j += 1

    return predicted_tumor_images


def load_dgl_graphs_from_bin(file_path, ids_path):
    dgl_graph_list, _ = dgl.load_graphs(file_path)
    with open(f'{ids_path}', 'rb') as file:
        ids = pickle.load(file)
    return dgl_graph_list, ids


def count_labels(triple_list):
    label_counts = {}
    labels = [triple[2] for triple in triple_list]

    # Concatenate all the arrays into one 
    all_labels = np.concatenate(labels) 

    # Count the occurrences of each label 
    counter = Counter(all_labels) 

    # Create a dict with the counts for labels 1 to 4 
    counts_dict = {i+1: counter[i+1] for i in range(4)} 

    return counts_dict


def class_weights_tensor(label_weights):
    num_classes = max(label_weights.keys())
    weight_tensor = torch.zeros(num_classes, dtype=torch.float32)

    # Sort the dictionary by keys (labels)
    sorted_label_weights = sorted(label_weights.items(), key=lambda x: x[0])

    for label, weight in sorted_label_weights:
        weight_tensor[label - 1] = weight  # Subtract 1 if your labels start from 1
    return weight_tensor


def compute_average_weights(graphs):
    print('computing weights...')
    label_counts = count_labels(graphs)
    total_count = sum(label_counts.values())
    class_weights = {label: total_count / count for label, count in label_counts.items()}
    weight_tensor = class_weights_tensor(class_weights)
    return weight_tensor


def DGLGraph_plotter(nx_graph):
    # Color Map
    color_map = {3: (0.5, 0.5, 0.5, 0.2), 1: 'blue', 2: 'yellow', 4: 'red'}

    # Assign color to nodes taking their label
    node_colors = [color_map[nx_graph.nodes[node]['label']] for node in nx_graph.nodes]

    # Prepare the list of colors based on the edge labels
    edge_colors = []
    for u, v in nx_graph.edges():
        if nx_graph.nodes[u]['label'] in {1, 2, 4} and nx_graph.nodes[v]['label'] in {1, 2, 4}:
            edge_colors.append('red')
        else:
            edge_colors.append((0, 0, 0, 0.2))  # Black color with 0.2 transparency

    plt.figure(figsize=(15, 11))
    # Draw the graph
    pos = nx.spring_layout(nx_graph)
    nx.draw(nx_graph, pos, node_color=node_colors, edge_color=edge_colors)

    # Create a dictionary with node keys for nodes with labels 1, 2, and 4
    label_dict = {node: node for node in nx_graph.nodes if nx_graph.nodes[node]['label'] in {1, 2, 4}}

    # Draw node labels
    nx.draw_networkx_labels(nx_graph, pos, labels=label_dict, font_size = 8)

    # Create the legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Enhanced', markerfacecolor='red', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='Edema', markerfacecolor='yellow', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='Necrosis', markerfacecolor='blue', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='Other', markerfacecolor=color_map[3], markersize=8)
    ]

    # Display the legend
    plt.legend(handles=legend_elements, loc='best')
    plt.show()


def predict(graph, feature, model):

    feature = torch.tensor(feature).float()
    
    # Use the model to get the output logits
    with torch.no_grad(): # Inference only
        logits = model(graph, feature)
    pred = logits.argmax(1)
    pred = pred + 1

    # pred[pred == 3] = 0
    
    return pred


def save_settings(timestamp, model, patience, lr, weight_decay, gamma, args_model, heads, residuals, \
                  val_dropout, layer_sizes, in_feats, n_classes, feat_drop, attn_drop, dataset_pickle_path, \
                  val_k, model_path = None):
    
    string_timestamp = timestamp.strftime("%Y%m%d-%H%M%S")
    if not model_path == None:
        os.makedirs(f'{metrics_testing_path}/{string_timestamp}')
        path = f'{metrics_testing_path}/{string_timestamp}/testing_'
    else:
        os.makedirs(f'{metrics_training_path}/{string_timestamp}')
        path = f'{metrics_training_path}/{string_timestamp}/training_'
    
    # Open the file in write mode ('w')
    with open(f'{path}{string_timestamp}_settings.txt', 'w') as f:
        f.write('-- DATASET PICKLE --\n')
        f.write(f'dataset pickle = {dataset_pickle_path}\n')
        
        f.write('-- MODEL PATH --\n')
        if not model_path == None:
            f.write(f'model path = {model_path} \n')

        f.write('\n-- TYPE MODEL --\n')
        f.write(f'model = {model}\n')

        f.write('\n-- HYPERPARAMS --\n')
        f.write(f'patience = {patience}\n')
        f.write(f'lr = {lr}\n')
        f.write(f'weight_decay = {weight_decay}\n')
        f.write(f'gamma = {gamma}\n')
        if args_model == 'GAT':
            f.write(f'heads = {heads}\n')
            f.write(f'residuals = {residuals}\n')
            f.write(f'val_feat_drop = {feat_drop}\n')
            f.write(f'val_attn_drop = {attn_drop}\n')
        elif args_model == 'GraphSage' or args_model == 'GIN':
            f.write(f'val_dropout = {val_dropout}\n')
        elif args_model == 'Cheb':
            f.write(f'val_dropout = {val_dropout}\n')
            f.write(f'val_k = {val_k}\n')

        f.write(f'layer_sizes = {layer_sizes}\n')

        # Write each variable on its own line
        f.write('\n-- PARAMETERS --\n')
        f.write(f'in_feats = {in_feats}\n')
        f.write(f'n_classes = {n_classes}\n')
        
        f.write('\n-- DATE --\n')
        f.write(f'timestamp = {timestamp}\n')


# Utilities for Unet refinement

DEFAULT_BACKGROUND_NODE_LOGITS = [[1.0,-1.0,-1.0,-1.0]]

def get_supervoxel_partitioning(mri_id):
    fp=f'{dataset_path}/BraTS2021_{mri_id}/BraTS2021_{mri_id}_supervoxels.nii.gz'
    return read_nifti(fp,np.int16)

def save_voxel_logits(mri_id,node_logits):
    global output_dir
    node_logits=node_logits.detach().cpu().numpy()
    supervoxel_partitioning = get_supervoxel_partitioning(mri_id)
    #add placeholder logits for healthy tissue
    node_logits = np.concatenate([node_logits,DEFAULT_BACKGROUND_NODE_LOGITS])
    voxel_logits = node_logits[supervoxel_partitioning]
    save_as_nifti(voxel_logits,f'{dataset_path}/BraTS2021_{mri_id}/BraTS2021_{mri_id}_logits.nii.gz')

def save_as_nifti(img,fp):
    affine_mat = np.array([
        [ -1.0,  -0.0,  -0.0,  -0.0],
        [ -0.0,  -1.0,  -0.0, 239.0],
        [  0.0,   0.0,   1.0,   0.0],
        [  0.0,   0.0,   0.0,   1.0],
        ])
    img = nib.nifti1.Nifti1Image(img, affine_mat)
    nib.save(img, fp)

def read_nifti(fp,data_type):
    nib_obj = nib.load(fp)
    return np.array(nib_obj.dataobj,dtype=data_type)

def save_splitted_logits(batched_logits, ids):
    start_index = 0
    for id in ids:
        G, features, labels, id = get_graph(f'{dataset_path}/BraTS2021_{id}/BraTS2021_{id}_nxgraph.json', id)
        num_nodes = len(G.nodes())

        # Extract logits for the current graph
        logits = batched_logits[start_index : start_index + num_nodes]

        # Save the logits
        save_voxel_logits(id, logits)

        # Update the starting index for the next graph
        start_index += num_nodes