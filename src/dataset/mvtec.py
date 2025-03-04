import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torchvision import transforms

from .self_sup_tasks import patch_ex

WIDTH_BOUNDS_PCT = {
    "bottle": ((0.03, 0.4), (0.03, 0.4)),
    "cable": ((0.05, 0.4), (0.05, 0.4)),
    "capsule": ((0.03, 0.15), (0.03, 0.4)),
    "hazelnut": ((0.03, 0.35), (0.03, 0.35)),
    "metal_nut": ((0.03, 0.4), (0.03, 0.4)),
    "pill": ((0.03, 0.2), (0.03, 0.4)),
    "screw": ((0.03, 0.12), (0.03, 0.12)),
    "toothbrush": ((0.03, 0.4), (0.03, 0.2)),
    "transistor": ((0.03, 0.4), (0.03, 0.4)),
    "zipper": ((0.03, 0.4), (0.03, 0.2)),
    "carpet": ((0.03, 0.4), (0.03, 0.4)),
    "grid": ((0.03, 0.4), (0.03, 0.4)),
    "leather": ((0.03, 0.4), (0.03, 0.4)),
    "tile": ((0.03, 0.4), (0.03, 0.4)),
    "wood": ((0.03, 0.4), (0.03, 0.4)),
}

INTENSITY_LOGISTIC_PARAMS = {
    "bottle": (1 / 12, 24),
    "cable": (1 / 12, 24),
    "capsule": (1 / 2, 4),
    "hazelnut": (1 / 12, 24),
    "metal_nut": (1 / 3, 7),
    "pill": (1 / 3, 7),
    "screw": (1, 3),
    "toothbrush": (1 / 6, 15),
    "transistor": (1 / 6, 15),
    "zipper": (1 / 6, 15),
    "carpet": (1 / 3, 7),
    "grid": (1 / 3, 7),
    "leather": (1 / 3, 7),
    "tile": (1 / 3, 7),
    "wood": (1 / 6, 15),
}

BACKGROUND = {
    "bottle": (200, 60),
    "screw": (200, 60),
    "capsule": (200, 60),
    "zipper": (200, 60),
    "hazelnut": (20, 20),
    "pill": (20, 20),
    "toothbrush": (20, 20),
    "metal_nut": (20, 20),
}

OBJECTS = [
    "bottle",
    "cable",
    "capsule",
    "hazelnut",
    "metal_nut",
    "pill",
    "screw",
    "toothbrush",
    "transistor",
    "zipper",
]

TEXTURES = ["carpet", "grid", "leather", "tile", "wood"]


class MVtecDataset(Dataset):
    """
    Custom Dataset class for loading and preprocessing the MVTec Anomaly Detection (MVTec AD) dataset.
    
    This dataset class supports both training and testing modes, including optional few-shot learning.
    It applies necessary transformations such as resizing, normalization, and tensor conversion.
    
    Attributes:
        mode (str): Dataset mode, either 'train' or 'test'.
        root_dir (str): Root directory of the MVTec dataset, containing subdirectories for each class.
        class_label (Optional[int]): Numerical class label assigned during training.
        few_shot_k (Optional[int]): Number of few-shot normal samples used during inference.
        transform (transforms.Compose): Image transformations applied to input images.
        transform_gt (transforms.Compose): Image transformations applied to ground truth masks (for test mode).
        classes (List[str]): List of class names found in the dataset.
        num_classes (int): Total number of classes in the dataset.
        label_encoder (LabelEncoder): Encoder for mapping class names to numerical labels (used in training).
        img_paths (List[str]): List of image file paths.
        gt_paths (List[str]): List of ground truth mask file paths (used in test mode).
        few_shots (List[str]): List of few-shot normal sample file paths (used in inference).
    """ 

    def __init__(
        self,
        root_dir: str,
        mode: str = "train",
        class_label: Optional[int] = None,
        few_shot_k: Optional[int] = None,
        transform: Optional[Any] = None,
    ) -> None:
        """
        Initializes the MVtecDataset.

        Args:
            root_dir (str): Root directory of the MVTec AD dataset.
            mode (str, optional): Dataset mode ('train' or 'test'). Defaults to 'train'.
            class_label (Optional[int], optional): Class label used for training. Defaults to None.
            few_shot_k (Optional[int], optional): Number of few-shot normal samples for inference. Defaults to None.
            transform (Optional[Any], optional): Custom torchvision transforms for images. Defaults to None.
        """
        self.mode = mode # train or test
        self.class_label = class_label
        self.root_dir = Path(root_dir).as_posix()
        self.few_shot_k = few_shot_k # few shot normal samples (only valid when inference step)
        self.transform = transforms.Compose([
            transforms.Resize(
                (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])
        self.transform_gt = transforms.Compose([
            transforms.Resize(
                (224, 224), interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.ToTensor(),
        ])

        self.classes = [
            item for item in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, item))
        ]
        self.num_classes = len(self.classes)
        
        # Encode class labels if training mode
        if self.class_label and self.mode == "train":
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(self.classes)

        if transform:
            self.transform = transform

        # Collect image and ground truth paths
        self.img_paths = []
        self.gt_paths = []
        self.few_shots = []

        for root, _, files in os.walk(root_dir):
            for file in files:
                file_path = Path(os.path.join(root, file)).as_posix()
                if f"{self.mode}" in file_path and "good" not in file_path and "png" in file:
                    self.img_paths.append(file_path)
                if self.mode == "test" and "ground_truth" in file_path and "png" in file:
                    self.gt_paths.append(file_path)
                if self.few_shot_k is not None and "train" in file_path and "good" in file_path and "png" in file:
                    if len(self.few_shots) < self.few_shot_k:
                        self.few_shots.append(file_path)

        if len(self.gt_paths) == 0:
            self.gt_paths = None

    def __len__(self) -> int:
        """
        Returns:
            int: The total number of samples in the dataset.
        """
        return len(self.img_paths)
    
    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """
        Fetches a dataset sample by index.

        Args:
            index (int): Index of the sample to fetch.

        Returns:
            Tuple[torch.Tensor, ...]:
                - Image tensor (C, H, W): The preprocessed input image.
                - Class label tensor (long, optional): Numerical class label (only in training mode).
                - Ground truth mask tensor (C, H, W, optional): Anomaly ground truth mask (only in test mode).
                - Few-shot normal image tensor (C, H, W, optional): Few-shot sample image (only in inference mode).
                - Anomaly type (str, optional): Type of anomaly (only in test mode).
                - Image path (str): Path of the input image.
                - Ground truth path (str, optional): Path of the ground truth mask (only in test mode).
        """
        img_path = self.img_paths[index]
        image = self.transform(Image.open(img_path).convert("RGB"))
        class_name = img_path.split("/")[-4]
        class_label = []
        
        # Assign class label if training
        if self.class_label: # only used when training
            class_label = self.label_encoder.transform([class_name]).item()
            class_label = torch.tensor(class_label, dtype=torch.long)
        
        if self.mode == "test":
            anomaly_type = img_path.split("/")[-2]
            gt_path = self.gt_paths[index]
            gt = self.transform_gt(Image.open(gt_path))

            if self.few_shot_k is not None:
                few_shot_samples = self.transform(Image.open(self.few_shots[0]).convert("RGB"))
                return image, gt, few_shot_samples, anomaly_type, img_path, gt_path
        
            return image, gt, anomaly_type, img_path, gt_path
        
        return image, class_label, img_path
    

class MVtecNsaDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        transform: Optional[Any] = None,
    ) -> None:
        """
        Args:
            root_dir (str): Root directory of the dataset.
            transform (Optional[Any]): Transformations to be applied to images. Defaults to resizing to 224x224.
            random_seed (Optional[int]): Random seed for reproducibility. Defaults to None.
        """
        self.root_dir = Path(root_dir).as_posix()
        self.transform = transforms.Resize(
            (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
        )
        self.norm_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        if transform:
            self.transform = transform

        self.paths = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                file_path = Path(os.path.join(root, file)).as_posix()
                if "train" in file_path and "good" in file_path and "png" in file:
                    self.paths.append(file_path)

        self.prev_idx = np.random.randint(len(self.paths))

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.paths)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """
        Fetches a dataset sample by index.

        Args:
            index (int): Index of the sample to fetch.

        Returns:
            - If training mode:
                Tuple[torch.Tensor, torch.Tensor, str]:
                    - Image tensor (C, H, W): The preprocessed input image.
                    - Class label tensor (long): Numerical class label.
                    - Image path (str): Path of the input image.
            
            - If testing mode:
                Tuple[torch.Tensor, torch.Tensor, str, str, str]:
                    - Image tensor (C, H, W): The preprocessed input image.
                    - Ground truth mask tensor (C, H, W): Anomaly ground truth mask.
                    - Anomaly type (str): Type of anomaly.
                    - Image path (str): Path of the input image.
                    - Ground truth path (str): Path of the ground truth mask.
            
            - If testing mode with few-shot learning:
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str, str]:
                    - Image tensor (C, H, W): The preprocessed input image.
                    - Ground truth mask tensor (C, H, W): Anomaly ground truth mask.
                    - Few-shot normal image tensor (C, H, W): Few-shot sample image.
                    - Anomaly type (str): Type of anomaly.
                    - Image path (str): Path of the input image.
                    - Ground truth path (str): Path of the ground truth mask.
        """
        img_path = self.paths[index]
        img_normal = self.transform(Image.open(img_path).convert("RGB"))
        class_name = img_path.split("/")[-4]
        unique_seed = index # seed for reproductivity

        self_sup_args = {
            "width_bounds_pct": WIDTH_BOUNDS_PCT.get(class_name),
            "intensity_logistic_params": INTENSITY_LOGISTIC_PARAMS.get(class_name),
            "num_patches": 2,  # if single_patch else NUM_PATCHES.get(class_name),
            "min_object_pct": 0,
            "min_overlap_pct": 0.25,
            "gamma_params": (2, 0.05, 0.03),
            "resize": True,
            "shift": True,
            "same": False,
            "mode": cv2.NORMAL_CLONE,
            "label_mode": "logistic-intensity",
            "skip_background": BACKGROUND.get(class_name),
            "random_seed": unique_seed
        }
        if class_name in TEXTURES:
            self_sup_args["resize_bounds"] = (0.5, 2)

        img_normal = np.asarray(img_normal)

        prev = Image.open(self.paths[self.prev_idx]).convert("RGB")
        if self.transform is not None:
            prev = self.transform(prev)
        prev = np.asarray(prev)

        img_abnormal, mask = patch_ex(img_normal, prev, **self_sup_args)
        mask = torch.tensor(mask[None, ..., 0]).float()
        mask[mask > 0.15], mask[mask <= 0.15] = 1, 0

        self.prev_idx = index

        img_normal = self.norm_transform(img_normal.copy())
        img_abnormal = self.norm_transform(img_abnormal.copy())

        if np.all(mask.numpy() == 0.0):
            img_abnormal = img_normal

        return img_normal, img_abnormal, mask, img_path

    def collate(
        self, instances: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]]
    ) -> Dict[str, List[torch.Tensor]]:
        images = []
        masks = []
        img_paths = []

        for instance in instances:
            images.append(instance[0])
            masks.append(torch.zeros_like(instance[2]))
            img_paths.append(instance[3])

            images.append(instance[1])
            masks.append(instance[2])
            img_paths.append(instance[3])

        return dict(images=images, masks=masks, img_paths=img_paths)