import argparse
import copy
import os
import random
from pathlib import Path
from pprint import pprint
import json
import torch.optim as optim
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

    # Required arguments
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        help="Datraset to train on (CIFAR10 or CIFAR100)",
    )

    parser.add_argument(
    "--model_name",
    required=True,
    type=str,
    help="VGG, ResNet or ViT ",
    )
   
    parser.add_argument(
        "--seed",
        required=True,
        type=int,
        help="Seed to train on"
    )

    parser.add_argument(
        "--num_epochs",
        required=True,
        type=int,
        help="Seed to train on"
    )

    parser.add_argument(
        "--save_name",
        required=True,
        type=str,
        help="Project Name"
    )

    parser.add_argument(
        "--model_config",
        required=True,
        type=str,
        help="Path to model config",
    )

    parser.add_argument(
        "--models_dir",
        required=False,
        type=str,
        default = "./models",
        help="Directory to save models to."
    )

    parser.add_argument(
        "--data_dir",
        required=False,
        type=str,
        default = "./data/cifar",
        help="Data directory (Change when not using CIFAR)"
    )

    # Optional arguments
    parser.add_argument(
        "--batch_size",
        required=False,
        type=int,
        default = 256,
        help="Batch size to train on"
    )

    parser.add_argument(
        "--learning_rate",
        required=False,
        type=int,
        default = 0.01,
        help="Learning rate"
    )


    parser.add_argument(
        "--optimizer",
        required=False,
        type=str,
        default = "SGD",
        help="Optimizer (SGD,Adam)"
    )

    parser.add_argument(
        "--momentum",
        required=False,
        type=int,
        default = 0.9,
        help="Value for momentum"
    )


    parser.add_argument(
        "--criterion",
        required=False,
        type=str,
        default = "Cross-entropy",
        help="Loss metric"
    )

    parser.add_argument(
        "--dropout",
        required=False,
        type=int,
        default = 0.0,
        help="Dropout  value"
    )

    parser.add_argument(
        "--weight_decay",
        required=False,
        type=int,
        default = 0.0,
        help="Weight decay value (suggested 1e-4)"
    )

    parser.add_argument(
        "--aug",
        required=False,
        type=bool,
        default = False,
        help="Augmentation (True/False)"
    )

    parser.add_argument(
        "--config_dir",
        required=False,
        type=str,
        default = "./configs",
        help="Config Directory"
    )

    parser.add_argument(
        "--vgg_config",
        required=False,
        type=str,
        default = "vgg_config.json",
        help="VGG config"
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
    # Benchmark is not optimised to be deterministic


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
        full_save_dir
    )
    print("-----------------")
    print(f"Model saved at: {full_save_dir}")
    print("-----------------")


def main(args,run_num=0):
    
    with open(args.model_config, "r") as file:
        config_model = yaml.safe_load(file)

    pprint(args)
    config_model = EasyDict(config_model) 

    args.models_dir = Path(args.models_dir) / args.dataset

    args.models_dir = (
        args.models_dir / args.save_name / str(args.seed)
    )

    args.config_dir = Path(args.config_dir)

    set_seed(args.seed)
    device = get_device()

    print("Loading model")

    if "CIFAR" in args.dataset:
        data_loader_manager = CIFAR_dataloader.DataLoaderManagerCIFAR(
            config=args,
            dataset_name=args.dataset,
            seed=args.seed,
        )
    elif "ImageNet" in args.dataset:
        data_loader_manager = IMAGENET_dataloader.DataLoaderManagerImageNet(
            config=args,
            dataset_name=args.dataset,
            seed=args.seed,
        )

    print(f"Loading: {args.dataset}:")

    train_dataloader, test_dataloader = data_loader_manager.get_dataloaders()

    if args.model_name.startswith("VGG"):
        model = vgg_model.VGG(
            vgg_name=args.model_name,
            dropout=args.dropout,
            num_classes=data_loader_manager.num_classes,
            vgg_config=args.config_dir / args.vgg_config,
        ).to(device)
    elif args.model_name.startswith("ResNet"):
        n = (config_model.depth - 2) // 6
        model = resnet_model.ResNet_cifar(resnet_model.BasicBlock, [n,n,n],data_loader_manager.num_classes,args.dropout).to(device)
    elif args.model_name.startswith("ViT"):

        model = vit_model.ViT(
        patch_height = config_model.patch_height,
        patch_width = config_model.patch_width,
        embedding_dims = config_model.embedding_dims,
        dropout = args.dropout,
        heads = config_model.heads,
        num_layers = config_model.num_layers,
        forward_expansion = config_model.forward_expansion,
        max_len = config_model.max_len,
        layer_norm_eps = config_model.layer_norm_eps,
        num_classes = data_loader_manager.num_classes,
        ).to(device)

    else:
        raise NotImplementedError(f"Model {args.model_name} not implemented")
    
    learning_rate = args.learning_rate
    momentum = args.momentum
    if 'Cross-entropy' in args.criterion:
        criterion = torch.nn.CrossEntropyLoss()
    elif 'MSE' in args.criterion:
        criterion = torch.nn.MSELoss
    else:
        raise ValueError("Only Cross-entropy and MSE are supported")
    if 'SGD' in args.optimizer:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    elif 'Adam' in args.optimizer:
         optimizer = optim.Adam(model.parameters(), lr=learning_rate, momentum=momentum)
    else:
        raise ValueError("Only SGD and Adam are supported")

    trainer = Trainer.Trainer(
        model = model,
        train_loader = train_dataloader,
        test_loader = test_dataloader, 
        optimizer = optimizer,
        criterion = criterion,
        device = device,
        n_epoch = args.num_epochs,
        n_classes = data_loader_manager.num_classes
    )

    print("Random initialisation")
    save_model(model, save_file_name="initialisation", save_dir=args.models_dir)
    training_sequence,model  = trainer.train()
    save_model(model, args.save_name, args.models_dir)
    with open(f'{args.models_dir}/{args.save_name}.json', 'w') as f:
        json.dump(training_sequence, f)
    print("Finished")
    


if __name__ == "__main__":
    args = options_parser()

    main(args)