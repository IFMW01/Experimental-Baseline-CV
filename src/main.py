import argparse
import copy
import os
import random
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
import yaml
from easydict import EasyDict

import dataloaders.cifar_dataloaders as CIFAR_dataloader
import dataloaders.imagenet_dataloaders as IMAGENET_dataloader
import trainer.trainer as Trainer
import models.vgg as vgg_model
import models.resnet as resnet_model
import models.vit as vit_model



def options_parser():
    parser = argparse.ArgumentParser(description="Arguments for creating model")
    parser.add_argument(
        "--config_file",
        required=True,
        type=str,
        help="Path to config file",
    )

    args = parser.parse_args()

    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"Found {torch.cuda.device_count()} CUDA devices")
        print(f"({torch.cuda.get_device_name(device)})")
        return device
    else:
        print("No CUDA devices found, using CPU")
        return "cpu"


def save_model(model, save_file_name, save_dir):

    if os.path.isdir(save_dir) == False:
        os.makedirs(save_dir)

    full_save_dir = save_dir / f"{save_file_name}.pth"
    torch.save(
        model.state_dict(),
        full_save_dir,
    )
    print("-----------------")
    print(f"Model saved at: {full_save_dir}")
    print("-----------------")


def main(seed=None, run_num=0):
    args = options_parser()

    with open(args.config_file, "r") as file:
        config = yaml.safe_load(file)

    config.update(vars(args))

    pprint(config)

    config = EasyDict(config)

    config.models_dir = Path(config.models_dir) / args.dataset

    config.models_dir = (
        config.models_dir / args.save_name / str(seed)
    )

    config.config_dir = Path(config.config_dir)

    config.seed = seed if seed is not None else config.seed

    set_seed(seed)
    device = get_device()

    print("Loading model")

    if config.dataset.contains("CIFAR"):
        data_loader_manager = CIFAR_dataloader(
            config=config,
            dataset_name=config.dataset,
            seed=seed,
        )
    elif config.dataset.contains("ImageNet"):
        data_loader_manager = IMAGENET_dataloader(
            config=config,
            dataset_name=config.dataset,
            seed=seed,
        )

    print(f"Loading: {config.dataset}:")

    train_dataloader, test_dataloader = data_loader_manager.get_dataloaders()

    if config.model_name.startswith("VGG"):
        model = vgg_model.VGG(
            vgg_name=args.model_name,
            dropout=args.dropout,
            num_classes=data_loader_manager.num_classes,
            vgg_config=config.config_dir / config.vgg_config,
        ).to(device)
    elif config.model_name.startswith("ResNet20"):
        if args.dataset == 'CIFAR100':
            num_classes = 100
        else:
            num_classes = 10
        depth = 20
        n = (depth - 2) // 6
        model = resnet_model.ResNet_cifar(resnet_model.BasicBlock, [n,n,n],num_classes,args.dropout).to(device)
    elif config.model_name.startswith("ViT"):

        if args.dataset == 'CIFAR100':
            num_classes = 100
        else:
            num_classes = 10

        model = vit_model.ViT(
        patch_height = 16,
        patch_width = 16,
        embedding_dims = 768,
        dropout = args.dropout,
        heads = 4,
        num_layers = 4,
        forward_expansion = 4,
        max_len = int((32*32)/(16*16)),
        layer_norm_eps = 1e-5,
        num_classes = num_classes,
        ).to(device)

    else:
        raise NotImplementedError(f"Model {config.model_name} not implemented")
    
    trainer = Trainer(
        config=config,
        model=model,
        device=device,
        num_classes=data_loader_manager.num_classes,
        seed=seed,
    )

    print("Random initialisation")
    save_model(model, save_file_name="initialisation", save_dir=config.models_dir)
    model, training_sequence = trainer.train()
    save_model(model, config.save_name, config.models_dir)
    


if __name__ == "__main__":
    args = options_parser()

    with open(args.config_file, "r") as file:
        config = yaml.safe_load(file)
    config.update(vars(args))
    seeds = config["seed"]
    # seeds = [43, 91, 17]
    for seed in seeds:
        main(seed)