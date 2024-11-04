import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.prune as prune
from torch.nn.utils import parameters_to_vector as Params2Vec
from torch.nn.utils import vector_to_parameters as VectorToParams
from torch.nn.utils.prune import _validate_pruning_amount, _validate_pruning_amount_init, _compute_nparams_toprune
from scipy.stats import entropy
from numpy.linalg import norm

def get_function(model,data_loader,num_classes):
    dict = {}
    for i in range(0,num_classes):
        dict[i] = []
    model.eval()
    for index, (image,lable) in enumerate(data_loader):
        with torch.no_grad():
            output = model(image)
            output = output.detach().numpy()[0]
            output = output.tolist()
            output = np.array(output)
            dict[lable.numpy()[0]].append(output)
    return dict
    

def l1_distance(softmax_1,softmax_2):
    softmax_1 = np.array(softmax_1)
    softmax_2 = np.array(softmax_2)
    l1_dist = np.sum(np.abs(softmax_1 - softmax_2), axis=1)
    return np.mean(l1_dist)

def actviation_distance(softmax_1, softmax_2):
    softmax_1 = torch.tensor(np.array(softmax_1))
    softmax_2 = torch.tensor(np.array(softmax_2))
    diff = torch.sqrt(torch.sum(torch.square(softmax_1 - softmax_2), axis = 1))
    return torch.mean(diff)

def JS_divergence(softmax_1, softmax_2):
    softmax_1 = torch.tensor(np.array(softmax_1))
    softmax_2 = torch.tensor(np.array(softmax_2))
    _softmax_1 = softmax_1 / norm(softmax_1, ord=1)
    _softmax_2 = softmax_2 / norm(softmax_2, ord=1)
    _diff = 0.5 * (_softmax_1 + __softmax_2)
    return (0.5 * (entropy(_softmax_1, _diff) + entropy(__softmax_2, _diff))).mean()
    

def cosine_similarity_func(softmax_1, softmax_2):
    softmax_1 = torch.tensor(np.array(softmax_1)).flatten()
    softmax_2 = torch.tensor(np.array(softmax_2)).flatten()
    return torch.nan_to_num(torch.clip(torch.dot(
        softmax_1, softmax_2
    ) / (
        torch.linalg.norm(softmax_1)
        * torch.linalg.norm(softmax_2)
    ),-1, 1),0)

def vectorise_model(model):
    return Params2Vec(model.parameters())


def cosine_similarity_weights(base_model, model_weights):
    base_vec = vectorise_model(base_model)
    model_vec = vectorise_model(model_weights)
    return torch.nan_to_num(torch.clip(torch.dot(
        base_vec, model_vec
    ) / (
        torch.linalg.norm(base_vec)
        * torch.linalg.norm(model_vec)
    ),-1, 1),0)

# amount is value between 0 and 1
def global_prune_without_masks(model, amount):
    parameters_to_prune = []
    for mod in model.modules():
        if hasattr(mod, "weight"):
            if isinstance(mod.weight, torch.nn.Parameter):
                parameters_to_prune.append((mod, "weight"))
        if hasattr(mod, "bias"):
            if isinstance(mod.bias, torch.nn.Parameter):
                parameters_to_prune.append((mod, "bias"))
    parameters_to_prune = tuple(parameters_to_prune)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    for mod in model.modules():
        if hasattr(mod, "weight_orig"):
            if isinstance(mod.weight_orig, torch.nn.Parameter):
                prune.remove(mod, "weight")
        if hasattr(mod, "bias_orig"):
            if isinstance(mod.bias_orig, torch.nn.Parameter):
                prune.remove(mod, "bias")
    return model

# amount is value between 0 and 1
def global_prune_with_masks(model, amount):
    parameters_to_prune = []
    for mod in model.modules():
        if hasattr(mod, "weight"):
            if isinstance(mod.weight, torch.nn.Parameter):
                parameters_to_prune.append((mod, "weight"))
        if hasattr(mod, "bias"):
            if isinstance(mod.bias, torch.nn.Parameter):
                parameters_to_prune.append((mod, "bias"))
    parameters_to_prune = tuple(parameters_to_prune)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    return model