import torch
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import parameters_to_vector as Params2Vec
from torch.nn.utils import vector_to_parameters as VectorToParams
from torch.nn.utils.prune import _validate_pruning_amount, _validate_pruning_amount_init, _compute_nparams_toprune
from torchmetrics.classification import MulticlassCalibrationError
from scipy.stats import entropy
from numpy.linalg import norm
import json

def get_param_count(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    return total_params

def evaluate(model,dataloader,criteron,num_classes,device):
    model.eval()
    results = {}
    model_loss = 0.0
    correct = 0
    total = 0
    ece = 0
    ece = MulticlassCalibrationError(num_classes, n_bins=15, norm='l1')
    for data, target in dataloader:
        with torch.no_grad():
            if data.device != device:
                data = data.to(device) 
            if target.device != device:
                target = target.to(device) 
            output = model(data)
            loss = criteron(output, target)
            ece.update(torch.softmax(output, dim=1),target)
            model_loss += loss.item()
            _, predicted = torch.max(torch.softmax(output), 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    results['model_loss'] = model_loss/len(dataloader)
    results['ece'] = ece.compute().item()
    results['accuracy'] = 100 * (correct / total) 
    return results

def get_device():
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"Found {torch.cuda.device_count()} CUDA devices")
        print(f"({torch.cuda.get_device_name(device)})")
        return device
    else:
        print("No CUDA devices found, using CPU")
        return "cpu"

def get_function(model,data_loader,num_classes,criteron,device):
    model.to(device)
    model.eval()
    model_loss = 0.0
    correct = 0
    total = 0
    ece = 0
    ece = MulticlassCalibrationError(num_classes, n_bins=15, norm='l1')
    dict = {}
    results = {}
    for i in range(0,num_classes):
        dict[i] = []
    for index, (image,lable) in enumerate(data_loader):
        image = image.to(device)
        lable = lable.to(device)
        with torch.no_grad():
            output = model(image)
            output_processed = output.cpu().numpy()[0]
            output_processed = output_processed.tolist()
            key = lable.cpu().numpy()[0]
            dict[key].append(output_processed)
            loss = criteron(output, lable)
            ece.update(torch.softmax(output, dim=1),lable)
            model_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += lable.size(0)
            correct += (predicted == lable).sum().item()
    results['model_loss'] = model_loss/len(data_loader)
    results['ece'] = ece.compute().item()
    results['accuracy'] = (100 * (correct / total))
    return dict,results

def load_json(path):
    with open(path) as f:
        dict = json.load(f)
    return dict

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

def l1_distance(softmax_1,softmax_2):
    softmax_1 = np.array(softmax_1)
    softmax_2 = np.array(softmax_2)
    l1_dist = np.sum(np.abs(softmax_1 - softmax_2), axis=1)
    return np.mean(l1_dist)

def actviation_distance(softmax_1, softmax_2):
    softmax_1 = torch.tensor(np.array(softmax_1))
    softmax_2 = torch.tensor(np.array(softmax_2))
    diff = torch.sqrt(torch.sum(torch.square(softmax_1 - softmax_2), axis = 1))
    return torch.mean(diff).detach().cpu().item()

def JS_divergence(softmax_1, softmax_2):
    softmax_1 = torch.tensor(np.array(softmax_1))
    softmax_2 = torch.tensor(np.array(softmax_2))
    _softmax_1 = softmax_1 / norm(softmax_1, ord=1)
    _softmax_2 = softmax_2 / norm(softmax_2, ord=1)
    _diff = 0.5 * (_softmax_1 + _softmax_2)
    return ((0.5 * (entropy(_softmax_1, _diff) + entropy(_softmax_2, _diff))).mean())
    

def cosine_similarity_func(softmax_1, softmax_2):
    softmax_1 = torch.tensor(np.array(softmax_1)).flatten()
    softmax_2 = torch.tensor(np.array(softmax_2)).flatten()
    return torch.nan_to_num(torch.clip(torch.dot(
        softmax_1, softmax_2
    ) / (
        torch.linalg.norm(softmax_1)
        * torch.linalg.norm(softmax_2)
    ),-1, 1),0).detach().cpu().item()

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