from typing import Optional

import timm
import torch.nn as nn


class SwinTransformer(nn.Module):
    """
    SwinTransformer class to initialize and use a Swin Transformer model.

    Args:
        model_name (Optional[str]): Name of the Swin Transformer model to use. Defaults to 'swin_large_patch4_window7_224.ms_in22k'.
        pretrained (bool): Whether to load pretrained weights. Defaults to False.
        features_only (bool): If True, extracts feature maps instead of performing classification. Defaults to True.
        num_classes (Optional[int]): Number of output classes for classification. If set to 0, the model will output image embeddings. Defaults to None.

    Raises:
        ValueError: If both 'features_only' and 'num_classes' are defined.

    """
    def __init__(
        self, 
        model_name: Optional[str] = None,
        pretrained: bool = False,
        features_only: bool = False,
        num_classes: Optional[int] = None,
    ):
        super().__init__()

        if features_only and num_classes is not None:
            raise ValueError("'features_only' and 'num_classes' cannot both be defined. Please set only one.")
        
        if model_name is None:
            model_name = 'swin_large_patch4_window7_224.ms_in22k'

        self.model = timm.create_model(
            model_name=model_name, 
            pretrained=pretrained,
            features_only=features_only, # True for feature map extraction
            num_classes=num_classes, # Optional number of classes for classification
        )

    def forward(self, x):
        x = self.model(x)
        return x