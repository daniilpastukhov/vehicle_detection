from torch import nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

__all__ = ['build_efficientnet_v2_s']


def build_efficientnet_v2_s():
    num_classes = 1
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    # replace 1000 classes classifier with 1 class classifier (for binary classification)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.LazyLinear(num_classes),
    )
    return model
