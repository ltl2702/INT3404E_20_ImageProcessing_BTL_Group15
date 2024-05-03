import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os


class MyDataset(Dataset):
    def __init__(self):
        self.images = []
        self.labels = []

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        #  to do
        return self.images[index], self.labels[index]

    def getdata(self, data_path, mode):
        # to do
        return self.images, self.labels

