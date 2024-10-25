import random
import numpy as np
import torch
from datasets import load_dataset
import torchvision.transforms as transforms


class DataLoaderManagerImageNet:
    def __init__(
        self,
        config,
        dataset_name: str,
        seed: int,
    ):
        self.config = config

        self.dataset_name = dataset_name
        if dataset_name == "Tiny-ImageNet":
            self.dataset = 'Maysee/tiny-imagenet'
            self.num_classes = 200
        elif dataset_name == "ImageNet":
            self.dataset = 'ILSVRC/imagenet-1k'
            self.num_classes = 1000
        else:
            raise ValueError("Only Tiny-ImageNet and ImageNet-1K supported")

        self.aug_transformations = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        self.base_transformations = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

        self.seed = seed

    def apply_transform_aug(self, example):
        example['image'] = self.aug_transformations(example['image'])
        return example    

    def apply_transform(self,example):
         example['image'] = self.base_transformations(example['image'])
         return example

    def get_dataloaders(self):
        g = torch.Generator()
        g.manual_seed(self.seed)

        print("Loading normal train data")
        train_dataset = load_dataset('self.dataset', split='train')
        test_dataset = load_dataset('self.dataset', split='test')
        if self.config.aug: 
             train_dataset = train_dataset.map(self.apply_transform_aug)
        else:
             train_dataset = train_dataset.map(self.apply_transform)
        test_dataset = test_dataset.map(self.apply_transform)    


        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            worker_init_fn=self.seed_worker,
            generator=g,
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            worker_init_fn=self.seed_worker,
            generator=g,
        )

        return train_dataloader, test_dataloader

    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)