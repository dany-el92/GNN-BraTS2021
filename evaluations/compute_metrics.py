import numpy as np
import nibabel as nib
from medpy.metric.binary import hd95
from utilities import load_networkx_graph, generate_tumor_segmentation_from_graph
from dotenv import load_dotenv
import os

load_dotenv()
dataset_path = os.getenv('DATASET_PATH')


HEALTHY = 3
EDEMA = 2
NET = 1
ET = 4

# Calculate nodewise Dice score for WT, CT, and ET for a single brain.
# Expects two 1D vectors of integers.
def calculate_node_dices(preds, labels):
    p, l = preds, labels

    wt_preds = np.where(p == HEALTHY, 0, 1)
    wt_labs = np.where(l == HEALTHY, 0, 1)
    wt_dice = calculate_dice_from_logical_array(wt_preds, wt_labs)

    ct_preds = np.isin(p, [NET, ET]).astype(int)
    ct_labs = np.isin(l, [NET, ET]).astype(int)
    ct_dice = calculate_dice_from_logical_array(ct_preds, ct_labs)

    at_preds = np.where(p == ET, 1, 0)
    at_labs = np.where(l == ET, 1, 0)
    at_dice = calculate_dice_from_logical_array(at_preds, at_labs)

    return [wt_dice, ct_dice, at_dice]


# Each tumor region (WT, CT, ET) is binarized for both the prediction and ground truth 
# and then the overlapping volume is calculated.
def calculate_dice_from_logical_array(binary_predictions, binary_ground_truth):
    true_positives = np.logical_and(binary_predictions == 1, binary_ground_truth == 1)
    false_positives = np.logical_and(binary_predictions == 1, binary_ground_truth == 0)
    false_negatives = np.logical_and(binary_predictions == 0, binary_ground_truth == 1)
    tp, fp, fn = np.count_nonzero(true_positives), np.count_nonzero(false_positives), np.count_nonzero(false_negatives)
    # The case where no such labels exist (only really relevant for ET case).
    if (tp + fp + fn) == 0:
        return 1
    return (2 * tp) / (2 * tp + fp + fn)


#calculates voxelwise WT,CT,ET Dice and HD95 for a single brain.
#Expects two n-D (only tested with 2D or 3D) arrays of integers
def calculate_brats_metrics(predicted_voxels,true_voxels):
    wt_preds = np.where(predicted_voxels==HEALTHY,0,1)
    wt_gt = np.where(true_voxels==HEALTHY,0,1)
    wt_dice = calculate_dice_from_logical_array(wt_preds,wt_gt)
    wt_hd = calculate_hd95_from_logical_array(wt_preds,wt_gt)

    ct_preds = np.isin(predicted_voxels,[NET,ET]).astype(int)
    ct_gt = np.isin(true_voxels,[NET,ET]).astype(int)
    ct_dice = calculate_dice_from_logical_array(ct_preds,ct_gt)
    ct_hd = calculate_hd95_from_logical_array(ct_preds,ct_gt)

    at_preds = np.where(predicted_voxels==ET,1,0)
    at_gt = np.where(true_voxels==ET,1,0)
    at_dice = calculate_dice_from_logical_array(at_preds,at_gt)
    at_hd = calculate_hd95_from_logical_array(at_preds,at_gt)

    return [wt_dice,ct_dice,at_dice,wt_hd,ct_hd,at_hd]


#wrapper around hd95 function that handles the case where one or more labels are missing from the ground truth or prediction.
def calculate_hd95_from_logical_array(pred,gt):
    try:
        hd = hd95(pred,gt)
    #no positive (1) voxels present in one of the inputs
    except RuntimeError as e:
        #then this label isnt present in either the prediction or gt, so assign a distance of zero since the pred was correct
        if(not 1 in pred and not 1 in gt):
            hd = 0
        #return maximal distance
        else:
            hd = 300
    finally:
        return hd
    

def voxel_wise_batch_score(predicted_labels, ids_list):
    splitted_labels_list = []
    predicted_tumor_segmentation_list = []
    start_nodes_counter = 0
    end_nodes_counter = 0
    # wt_dice_array = np.array([])
    # ct_dice_array = np.array([])
    # at_dice_array = np.array([])
    # wt_hd_array = np.array([])
    # ct_hd_array = np.array([])
    # at_hd_array = np.array([])

    matrix = np.empty((0,6), int)

    for id in ids_list:
        json_graph = load_networkx_graph(f'{dataset_path}/BraTS2021_{id}/BraTS2021_{id}_nxgraph.json')
        num_nodes = len(json_graph.nodes())
        
        end_nodes_counter = num_nodes + start_nodes_counter

        splitted_labels = predicted_labels[start_nodes_counter:end_nodes_counter]
        splitted_labels_list.append(splitted_labels)

        start_nodes_counter = end_nodes_counter

        slic = nib.load(f'{dataset_path}/BraTS2021_{id}/BraTS2021_{id}_supervoxels.nii.gz').get_fdata()
        
        predicted_tumor_segmentation = generate_tumor_segmentation_from_graph(splitted_labels, slic)
        # predicted_tumor_segmentation_list.append(predicted_tumor_segmentation)
        
        wt_dice, ct_dice, at_dice, wt_hd, ct_hd, at_hd = calculate_dice_hd95(predicted_tumor_segmentation, id)
        
        row_scores = np.array([wt_dice, ct_dice, at_dice, wt_hd, ct_hd, at_hd])
        matrix = np.vstack([matrix, row_scores])

    mean_scores = np.mean(matrix, axis=0)
    
        # np.append(wt_dice_array, wt_dice)
        # np.append(ct_dice_array, ct_dice)
        # np.append(at_dice_array, at_dice)
        # np.append(wt_hd_array, wt_hd)
        # np.append(ct_hd_array, ct_hd)
        # np.append(at_hd_array, at_hd)

    return mean_scores


def calculate_dice_hd95(predicted_tumor_segmentation, id):
    fp = nib.load(f'{dataset_path}/BraTS2021_{id}/BraTS2021_{id}_label.nii.gz')
    label = np.array(fp.dataobj,dtype=np.int16)

    label[label == 3] = 4 # Trasforma le etichette 3 in 4
    label[label== 2] = -1   # Assegna temporaneamente il valore -1 alle etichette 2
    label[label == 1] = 2   # Trasforma le etichette 1 in 2
    label[label == -1] = 1  # Trasforma le etichette temporanee (-1) in 1
    label[label == 0] = 3  # Trasforma le etichette 0 in 3

    wt_dice, ct_dice, at_dice, wt_hd, ct_hd, at_hd = calculate_brats_metrics(predicted_tumor_segmentation, label)
    
    return wt_dice, ct_dice, at_dice, wt_hd, ct_hd, at_hd


def compute_ASA(supervoxels, ground_truth):
    regions = [1, 2, 4]
    ASA_scores = {}

    for region in regions:
        # Get the supervoxels and ground truth for the current region
        region_supervoxels = (supervoxels == region)
        region_ground_truth = (ground_truth == region)

        # Compute the intersection of the supervoxels and ground truth
        intersection = np.logical_and(region_supervoxels, region_ground_truth)

        # Compute ASA for the current region
        numerator = np.sum(np.max(intersection, axis=(1,2,3)))
        denominator = np.sum(region_ground_truth)
        ASA = numerator / denominator

        ASA_scores[region] = ASA

    return ASA_scores
