import random
import numpy as np
import torch
import torchvision.datasets as cifar_datasets
import torchvision.transforms as transforms
# from autoaugment import CIFAR10Policy

class DataLoaderManagerCIFAR:
    def __init__(
        self,
        config,
        dataset_name: str,
        seed: int,
    ):
        self.config = config

        self.dataset_name = dataset_name
        if dataset_name == "CIFAR10":
            self.dataset = cifar_datasets.CIFAR10
            self.num_classes = 10
        elif dataset_name == "CIFAR100":
            self.dataset = cifar_datasets.CIFAR100
            self.num_classes = 100
        else:
            raise ValueError("Only CIFAR10 and CIFAR100 supported")

        self.aug_transformations = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4, fill=128), # fill parameter needs torchvision installed from source
                transforms.RandomHorizontalFlip(), 
                # CIFAR10Policy(), 
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        self.base_transformations = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        self.seed = seed

    def get_dataloaders(self):
        g = torch.Generator()
        g.manual_seed(self.seed)

        # if self.config.adversarial == True and self.config.baseline == True:
        #     print("Loading random label train data")
        #     train_dataset = cifar_random_lables.get_random_cifar_dataset(
        #         self.dataset,
        #         self.num_classes,
        #         corrupt_prob=1.0,
        #         root=self.config.cifar_dir,
        #         download=True,
        #         transform=self.aug_transformations if self.config.aug else self.base_transformations,
        #         train=True,
        #     )
        # else:
        print("Loading normal train data")
        train_dataset = self.dataset(
            root=self.config.cifar_dir,
            train=True,
            download=True,
            transform=self.aug_transformations if self.config.aug else self.base_transformations,
        )

        eval_dataset = self.dataset(
            root=self.config.cifar_dir,
            train=False,
            download=True,
            transform=self.base_transformations,
        )

        num_eval = len(eval_dataset)
        indices = list(range(num_eval))
        split = num_eval - 5000
        dev_idx, test_idx = indices[:split], indices[split:]

        # dev_dataset = torch.utils.data.Subset(eval_dataset, dev_idx)
        # test_dataset = torch.utils.data.Subset(eval_dataset, test_idx)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            worker_init_fn=self.seed_worker,
            generator=g,
        )

        test_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
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

# Taken from https://github.com/pluskid/fitting-random-labels/blob/master/cifar10_data.py

class CIFARRandomLabelsBase:
    """CIFAR10 dataset, with support for randomly corrupt labels.
    Params
    ------
    num_classes: int
      The number of classes in the dataset.
    corrupt_prob: float
      Default 0.0. The probability of a label being replaced with
      random label.
    """

    def __init__(self, num_classes, corrupt_prob=0.0, **kwargs):
        super(CIFARRandomLabelsBase, self).__init__(**kwargs)
        self.n_classes = num_classes
        if corrupt_prob > 0:
            self.corrupt_labels(corrupt_prob)

    def corrupt_labels(self, corrupt_prob):
        labels = np.array(self.targets)
        np.random.seed(12345)
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(self.n_classes, mask.sum())
        labels[mask] = rnd_labels
        # we need to explicitly cast the labels from npy.int64 to
        # builtin int type, otherwise pytorch will fail...
        labels = [int(x) for x in labels]

        self.targets = labels


def get_random_cifar_dataset(dataset, num_classes, corrupt_prob=0.0, **kwargs):

    class CIFARRandomLabels(CIFARRandomLabelsBase, dataset):
        pass

    return CIFARRandomLabels(num_classes, corrupt_prob, **kwargs)