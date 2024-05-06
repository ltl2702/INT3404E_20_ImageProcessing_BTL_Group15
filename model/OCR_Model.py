import torch.nn as nn
import config

# N=(W-F+2P)/S+1
class OCR_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            # size = 128*128*3
            # Conv1
            nn.Conv2d(3, 32, 5, padding=2),  # size = 128*128*32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # size = 64*64*32
            # Conv2
            nn.Conv2d(32, 64, 5, padding=2),  # size = 64*64*64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # size = 32*32*64
            # Conv3
            nn.Conv2d(64, 128, 5, padding=2),  # size = 32*32*128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # size = 16*16*128
            # Conv4
            nn.Conv2d(128, 256, 5, padding=2),  # size = 16*16*256
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # size = 8*8*256
            # Conv5
            nn.Conv2d(256, 256, 5, padding=2),  # size = 8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # size = 4*4*256  
            # Conv6
            nn.Conv2d(256, 512, 5, padding=2),  # size = 4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # size = 2*2*512
        )
        self.dense = nn.Sequential(
            # FC1
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.25),
            # FC2
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.25),
            # FC3
            nn.Linear(1024, config.number_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        return x
