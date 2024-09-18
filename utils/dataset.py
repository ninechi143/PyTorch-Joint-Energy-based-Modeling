# -*- coding: utf-8 -*-

import torch

from torch.utils.data import Dataset
import torchvision

import os
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
import glob
from PIL import Image
import cv2
import csv


def collate_fn(batch):
    batch = list(filter(lambda x: (x is not None) and (None not in x), batch))
    if len(batch) == 0: 
        return torch.Tensor()
    return torch.utils.data.dataloader.default_collate(batch)



class base_dataset(Dataset):

    def __init__(self):

        self.transform = None
        self.data_list = []
        self.n_samples = 0
     
    def collect_data(self, *args, **kwargs):
        pass


    def set_transform(self , transform = None):    
        self.transform = transform
    

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        pass


    def __len__(self):
        return self.n_samples



class pretrain_dataset(base_dataset):

    def __init__(self, dataset_path, stage):

        super().__init__()
        self.stage = stage
        self.data_list , self.n_samples = self.collect_data(dataset_path)


    def collect_data(self, dataset_path):

        data_list = []

        dataset_path = os.path.join(dataset_path, self.stage)
        for dataset in tqdm(os.listdir(dataset_path), desc = f"{self.stage}"):

            png_list = list(Path(os.path.join(dataset_path, dataset)).rglob("*.png"))
            data_list += png_list

        assert len(data_list) != 0, "No any data in database, please check the argument of --data or codes of dataset.py"

        print(f"Stage: {self.stage}, #data = {len(data_list)}")

        return data_list, len(data_list)



    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        
        file_path = self.data_list[index]

        data = np.array(Image.open(file_path).convert("L")).astype(np.float32) # here, the value range is from 0 to 255
        data = torch.unsqueeze(torch.from_numpy(data), dim = 0).float() / 255.0

        # label = np.array(data)
        # label =  torch.unsqueeze(torch.from_numpy(label), dim = 0).float() / 255.0

        if self.transforms:
            data = self.transforms(data)
            # label = self.transforms(label)

        return data, data


class downstream_task_dataset(base_dataset):
    def __init__(self, train_stage, transform,):

        super().__init__()
        self.train_stage = train_stage
        self.transform = transform

        cifar10 = torchvision.datasets.CIFAR10(root="./data", transform=None, download=True, train=self.train_stage)
        
        # self.data = np.array(cifar10.data)[:int(len(cifar10) * 0.1)]
        # self.targets = np.array(cifar10.targets)[:int(len(cifar10) * 0.1)]
        self.data = np.array(cifar10.data)
        self.targets = np.array(cifar10.targets)
        self.n_samples = self.data.shape[0]

        
    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):

        data, target = self.data[index], self.targets[index]

        if data is None or target is None:  
            return None, None

        if self.transform:
            data = self.transform(data)
        else:
            data = torch.from_numpy(data / 255.).float()

        target = torch.from_numpy(np.array(target)).long() #torch.CrossEntropy loss class requires scalar-target (not one-hot) and long-integer type
        return data, target
    

if __name__ == "__main__":

    # a = torchvision.datasets.MNIST(root = "./data", train = True, transform=None, download=True)
    # b = torchvision.datasets.MNIST(root = "./data", train = False, transform=None)

    # ax = a.data.numpy()
    # ay = a.targets.numpy()

    # print(type(ax) , type(ay), ax.shape, ay.shape)     
     
    # # print(ay[5])   # not yet one-hot coding, so it is just a single number
    # # print(ax[5])   # not yet normalizer, so the range is from 0 to 255

    
    import torchvision.transforms as tt
    transforms_train = tt.Compose( [
                                # tt.Lambda(lambda x: 2. * (np.array(x) / 255.) - 1.),
                                # tt.Lambda(lambda x: torch.from_numpy(x).float()),
                                # tt.Lambda(lambda x: torch.permute(x, (2,0,1))),
                                tt.ToTensor(),
                                tt.Pad(4, padding_mode="reflect"),
                                tt.RandomCrop(32),
                                tt.RandomHorizontalFlip(),
                                tt.Normalize((.5, .5, .5), (.5, .5, .5)),
                                tt.Lambda(lambda x: x + 0.02 * torch.randn_like(x))
                                ])

    dataset = downstream_task_dataset(train_stage = True, transform = transforms_train, OOD = True)
    print(len(dataset), dataset[0][0].shape, dataset[0][1].shape)
    print("Done.")