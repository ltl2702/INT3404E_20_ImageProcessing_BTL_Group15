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
        img, label = self.images[index], self.labels[index]
        img = Image.open(img).convert("RGB")
        img = img.resize((128,128), resample=Image.Resampling.BILINEAR)

        # ImageNet推荐的RGB通道均值、方差
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        img = tf(img)
        label = torch.tensor(label)
        return img, label

    def getdata(self, data_path, mode):
        dict_path = os.path.join(data_path, mode + "_labels.csv")
        img_root = os.path.join(data_path, mode + "/images")
        content = pd.read_csv(dict_path)
        image_names = content["image_name"]
        label = content["label"]
        labels = []
        image_paths =  []
        for i in range(len(content)):
            if mode == "train":
                file_path = os.path.join(str(label[i]),image_names[i])
            if mode == "val":
                file_path = image_names[i] + ".jpg"
            labels.append(label[i])
            image_paths.append(os.path.join(img_root, file_path))
        
        self.images = image_paths
        self.labels = labels

