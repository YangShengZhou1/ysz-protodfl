import copy
import logging

import numpy as np
import torch

def fake_prune(each_prune_ratio, param_dict, mask):
    '''
    This function derives the new pruning mask, it put 0 for the weights under the given percentile

    :param each_prune_ratio: pruning ratio
    :param param_dict: a dictionary containing model parameters
    :param mask: the pruning mask

    :return mask: updated pruning mask
    '''
    new_masks = copy.deepcopy(mask)
    for name in param_dict:
        if "weight" in name and "bn" not in name:
            tensor = param_dict[name]
            mask_tensor = mask[name]
            alive = tensor[torch.nonzero(tensor * mask_tensor, as_tuple=True)]  # flattened array of nonzero values
            percentile_value = torch.quantile(alive.abs(), each_prune_ratio).item()
            new_mask = torch.where(tensor.abs() < percentile_value, torch.tensor(0, device=tensor.device), mask_tensor)
            new_masks[name] = new_mask
    return new_masks

def real_prune(param_dict, mask):
    '''
    This function applies the derived mask. It zeros the weights needed to be pruned based on the updated mask

    :param param_dict: a dictionary containing model parameters
    :param mask: pruning mask

    :return state_dict: updated (pruned) model state_dict
    '''
    new_param_dict = copy.deepcopy(param_dict)
    for name in param_dict:
        param = param_dict[name]
        weight_dev = param.device
        new_param_dict[name] = param * mask[name].to(weight_dev)
    return new_param_dict

def dist_masks(m1, m2):
    '''
    Calculates hamming distance of two pruning masks. It averages the hamming distance of all layers and returns it

    :param m1: pruning mask 1
    :param m2: pruning mask 2

    :return average hamming distance of two pruning masks:
    '''
    temp_dist = []
    for name in m1:
        m1_tensor = m1[name].to(torch.bool)
        m2_tensor = m2[name].to(torch.bool)
        hamming_dist = (m1_tensor != m2_tensor).float().mean().item()
        temp_dist.append(hamming_dist)
    dist = np.mean(temp_dist)
    return dist

def print_pruning(param_dict, is_print=False):
    '''
    This function prints the pruning percentage and status of a given model

    :param param_dict: a dictionary containing model parameters
    :param is_print: flag to indicate if printing is required

    :return pruning percentage, number of remaining weights:
    '''
    nonzero = 0
    total = 0
    for name in param_dict:
        tensor = param_dict[name]
        nz_count = torch.count_nonzero(tensor)
        total_params = torch.numel(tensor)
        nonzero += nz_count
        total += total_params
        if is_print:
            print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total},'
                  f'Compression rate : {total / nonzero:10.2f}x  ({100 * (total - nonzero) / total:3.2f}% pruned)')
    return nonzero / total, nonzero
