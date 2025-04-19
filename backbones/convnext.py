import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, convnext_small, convnext_base, convnext_large

class ArcIrisConvNeXt(nn.Module):
    def __init__(self, model_name: str, num_features=512, fp16=False, **kwargs):
        super().__init__()
        self.fp16 = fp16

        if model_name == "convnext_tiny":
            self.backbone = convnext_tiny(**kwargs)
        elif model_name == "convnext_small":
            self.backbone = convnext_small(**kwargs)
        elif model_name == "convnext_base":
            self.backbone = convnext_base(**kwargs)
        elif model_name == "convnext_large":
            self.backbone = convnext_large(**kwargs)
        else:
            raise ValueError("Unsupported ConvNeXt model")

        self.fc = torch.nn.Linear(self.backbone.features[-1][-1].block[-2].out_features, num_features)
        self.flatten = torch.nn.Flatten(start_dim=1, end_dim=-1)
        self.features = torch.nn.BatchNorm1d(num_features, eps=1e-05)

    def forward(self, x):
        with torch.amp.autocast("cuda", enabled=self.fp16):
            x = self.backbone.features(x)
            x = self.backbone.avgpool(x)
            x = self.flatten(x)
        x = self.fc(x.float() if self.fp16 else x)
        x = self.features(x)
        return x
