import argparse
from pathlib import Path
from typing import Tuple

import cv2
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F

from models.swin_transformer import SwinTransformer


class FewShotADLit(L.LightningModule):
    """
    This module is designed for few-shot anomaly detection tasks using Swin Transformer-based feature extraction.
    
    Features:
    - Implements the `predict_step` method for anomaly detection.
    - Computes similarity maps between query images and few-shot normal samples.
    - Saves the similarity maps as images for visualization.
    
    Attributes:
        args (Dict[str, Any]): Configuration arguments.
        encoder (SwinTransformer): Swin Transformer-based feature extractor.
    """
    def __init__(self, args: argparse.Namespace):
        super(FewShotADLit, self).__init__()
        self.args = args
        self.encoder = SwinTransformer(
            model_name=args.model_name,
            pretrained=True,
            features_only=True
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to extract feature representations.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
        
        Returns:
            torch.Tensor: Extracted feature representations. num_layers x (B, C, H, W)
        """
        features = self.encoder(x)
        return features

    def predict_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        """
        Performs the prediction step for anomaly detection.

        Args:
            batch (Tuple[torch.Tensor, ...]): Input batch containing query images and few-shot normal samples.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Computed similarity map for anomaly detection.
        """
        anomaly_type = batch[3][0]
        class_name = batch[4][0].split("/")[-4]
        file_name = batch[4][0].split("/")[-1]

        save_dir = Path(self.args.save_dir) / class_name / anomaly_type
        save_dir.mkdir(parents=True, exist_ok=True)

        query_inputs = batch[0]
        normal_inputs = batch[2]

        query_patches = self.encoder(query_inputs) # L x [B, H, W, D]
        normal_patches = self.encoder(normal_inputs) # L x [B, H, W, D]

        similarity_map = self.get_similarity_map(query_patches, normal_patches)
        similarity_map = similarity_map.squeeze(0) # suppose batch size = 1
        similarity_map_np = self.tensor_to_np(similarity_map)
        similarity_map_np = (similarity_map_np * 255).astype(np.uint8)

        cv2.imwrite(str(save_dir / file_name), similarity_map_np)

        return similarity_map

    def get_similarity_map(self, query_patches, normal_patches):
        """
        Computes a similarity map between query and normal patches.

        Args:
            query_patches (torch.Tensor): Feature patches from query images.
            normal_patches (torch.Tensor): Feature patches from normal images.
        
        Returns:
            torch.Tensor: Computed similarity map of shape (B, 1, 224, 224).
        """
        sims = []

        for i in range(len(query_patches)):
            B, H, W, C = query_patches[i].shape
            query_patches_tokens = query_patches[i].view(B, H*W, 1, C)
            normal_patches_tokens = normal_patches[i].reshape(B, 1, -1, C)
            cosine_similarity_matrix = F.cosine_similarity(query_patches_tokens, normal_patches_tokens, dim=-1)
            sim_max, _ = torch.max(cosine_similarity_matrix, dim=-1)
            sims.append(sim_max)
        
        max_resolution = sims[0].shape[-1]
        resized_sims = [
            F.interpolate(sim.view(B, 1, int(sim.shape[1]**0.5), int(sim.shape[1]**0.5)), 
                        size=(int(max_resolution**0.5), int(max_resolution**0.5)), 
                        mode="bilinear", 
                        align_corners=False).view(sim.shape[0], -1)
            for sim in sims
        ]

        sim = torch.mean(torch.stack(resized_sims, dim=0), dim=0).reshape(B, 1, 56, 56)
        sim = F.interpolate(sim, size=224, mode='bilinear', align_corners=True)
        similarity_map = 1 - sim
        
        return similarity_map.cpu()
    
    @staticmethod
    def tensor_to_np(tensor: torch.Tensor) -> np.ndarray:
        """
        Converts a PyTorch tensor to a NumPy array.

        Args:
            tensor (torch.Tensor): A PyTorch tensor with shape [C, H, W].

        Returns:
            np.ndarray: A NumPy array with shape [H, W, C] normalized to range [0, 1].
        """
        tensor = tensor.detach().cpu().numpy()
        tensor = np.transpose(tensor, (1, 2, 0))  # [C, H, W] to [H, W, C]
        if not np.all(tensor == 0.0):
            tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        return tensor