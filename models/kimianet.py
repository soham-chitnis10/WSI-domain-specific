import torch
import torchvision.models as models
import torch.nn as nn
import pytorch_lightning as pl

class KimiaNet(nn.Module):
    def __init__(self, imagenet=False):
        super(KimiaNet, self).__init__()
        self.model = models.densenet121(pretrained=True)
        self.model =  nn.Sequential(self.model.features , nn.AdaptiveAvgPool2d(output_size= (1,1)),nn.Flatten(1,-1))
        if not imagenet:
            self.model.load_state_dict(torch.load('saved_models/KimiaNet.pth'))

    def forward(self, x):
        feature = self.model(x)
        return feature

    
