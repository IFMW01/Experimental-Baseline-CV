import torch
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector as Params2Vec
from torch.nn.utils import vector_to_parameters as VectorToParams
from torch.nn.utils.prune import _validate_pruning_amount, _validate_pruning_amount_init, _compute_nparams_toprune

def actviation_distance(softmax_1, softmax_2):
    diff = torch.sqrt(torch.sum(torch.square(softmax_1 - softmax_2), axis = 1))
    return diff

# JS DIST Metric Calculation
def JS_divergence(softmax_1, softmax_2):
    diff = (softmax_1+softmax_2)/2 
    js = (0.5*F.kl_div(torch.log(softmax_1), diff) + 0.5*F.kl_div(torch.log(softmax_2), diff)).detach().cpu().item()
    return js

def vectorise_model(model):
    return Params2Vec(model.parameters())

def cosine_similarity(base_weights, model_weights):
    return torch.nan_to_num(torch.clip(torch.dot(
        base_weights, model_weights
    ) / (
        torch.linalg.norm(base_weights)
        * torch.linalg.norm(model_weights)
    ),-1, 1),0)

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