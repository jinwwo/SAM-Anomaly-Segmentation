from typing import List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import morphology


def separate_objects(mask: np.array) -> List[np.array]:
    """
    Separate distinct objects in a binary mask.

    Args:
        mask (np.array): Input mask where objects are represented as nonzero values.

    Returns:
        List[np.array]: A list of binary masks, each representing a separate object.
    """
    _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    object_masks = []

    for contour in contours:
        object_mask = np.zeros_like(mask)
        cv2.drawContours(object_mask, [contour], -1, 255, thickness=cv2.FILLED)
        object_masks.append(object_mask)

    return object_masks


def create_bbox_prompt(
    masks: List[np.array],
    mode_bbox: Optional[str] = None,
    padding: int = 0,
) -> List[List[int]]:
    """
    Generate bounding boxes for given object masks.

    Args:
        masks (List[np.array]): List of binary masks representing objects.
        mode_bbox (Optional[str]): If "naive", returns a single bounding box covering the entire image.
        padding (int): Additional padding added to the bounding box coordinates.

    Returns:
        List[List[int]]: A list of bounding boxes in [x_min, y_min, x_max, y_max] format.
    """
    bboxes = []
    if mode_bbox == "naive":
        h, w = masks[0].shape
        bboxes.append([0, 0, w, h])
        return np.array(bboxes)

    for mask in masks:
        nonzero_idx = np.nonzero(mask)
        upper = np.min(nonzero_idx[0]) + padding
        lower = np.max(nonzero_idx[0]) + padding
        left = np.min(nonzero_idx[1]) + padding
        right = np.max(nonzero_idx[1]) + padding
        bboxes.append([left, upper, right, lower])

    return np.array(bboxes)


def create_point_centroid(masks):
    """
    Compute the centroid of each object mask.

    Args:
        masks (List[np.array]): List of binary masks representing objects.

    Returns:
        Tuple[np.array, np.array]: 
            - Centroid points in (x, y) format.
            - Labels indicating valid points (all set to 1).
    """
    from scipy.ndimage import center_of_mass

    points = []
    for mask in masks:
        mask_points = []
        y, x = center_of_mass(mask)
        mask_points.append([int(x), int(y)])
        points.append(mask_points)

    p, q = np.array(points).shape[:2]
    labels = np.ones((p, q))  # (num_masks, num_points / 2)

    if len(masks) == 1:
        return np.array(points).squeeze(0), labels.squeeze(0)

    return np.array(points), labels


def create_point_prompt(masks, mode_point, n_points):
    if mode_point == "point":
        return create_point(masks)
    elif mode_point == "point_centroid":
        return create_point_centroid(masks)
    elif mode_point == "points":
        return create_points(masks, n_points)
    else:
        return (None, None)


def create_point(masks: List[np.array]) -> List[List[int]]:
    points = []
    for mask in masks:
        nonzero_idx = np.nonzero(mask)
        sampled_idx = int(len(nonzero_idx[0]) / 2)
        x, y = nonzero_idx[1][sampled_idx], nonzero_idx[0][sampled_idx]
        points.append([x, y])
    labels = [1] * len(points)  # the number of anomaly region
    
    return np.array(points), np.array(labels)


def modeset(mode: str, n_points: int):
    if mode == "naive":
        print("Mode set to [naive]. Using only bbox that covers the whole image.")
        mode_bbox = "naive"
        mode_point = None
    elif mode == "b":
        print("Mode set to [box (b)]. Using only bbox.")
        mode_bbox = "box"
        mode_point = None
    elif mode == "bp":
        print("Mode set to [box with 1 point (bp)].")
        mode_bbox = "box"
        mode_point = "point_centroid"
    else:
        print(f"Mode set to [box with {n_points} points (bps)].")
        mode_bbox = "box"
        mode_point = "points"

    return mode_bbox, mode_point


def create_points(masks: List[np.array], num_points_per_mask: int = 10):
    """
    Sample a fixed number of points from the foreground of each mask.

    Args:
        masks (List[np.array]): List of binary masks representing objects.
        num_points_per_mask (int): Number of points to sample per mask.

    Returns:
        Tuple[np.array, np.array]: 
            - Sampled points in (x, y) format.
            - Labels indicating valid points (all set to 1).
    """
    # need to use squeeze(0) when len(masks) == 1
    all_points = []  # (num_masks, num_points_per_mask, 2)
    for mask in masks:
        mask_points = []
        nonzero_idx = np.nonzero(mask)  # coords of white region
        total_points = len(nonzero_idx[0])
        if total_points == 0:
            continue

        # sampling on white region
        indices = np.linspace(0, total_points - 1, num_points_per_mask, dtype=int)
        for i in indices:
            y = nonzero_idx[0][i]
            x = nonzero_idx[1][i]
            mask_points.append([x, y])
        all_points.append(mask_points)

    p, q = np.array(all_points).shape[:2]
    labels = np.ones((p, q))  # (num_masks, num_points / 2)

    if len(masks) == 1:
        return np.array(all_points).squeeze(0), labels.squeeze(0)

    return np.array(all_points), labels


def mask_postprocessing(mask: np.array, kernel_size: int = 4) -> np.array:
    """
    Apply morphological opening to refine the mask.

    Args:
        mask (np.array): Input binary mask.
        kernel_size (int): Size of the morphological structuring element.

    Returns:
        np.array: Post-processed mask.
    """
    kernel = morphology.disk(kernel_size)
    mask = morphology.opening(mask, kernel)
    return mask * 255


def get_masks_with_bbox(mask, bboxes):
    anotated_mask = cv2.merge([mask, mask, mask])

    for bbox in bboxes:
        left, top, right, bottom = bbox
        cv2.rectangle(anotated_mask, (left, top), (right, bottom), (0, 255, 0), 2)

    return anotated_mask


def get_masks_with_point(mask, points):
    anotated_mask = cv2.merge([mask, mask, mask]) if len(mask.shape) == 2 else mask
    if len(points.shape) == 3:
        points = points.reshape(-1, 2)

    for point in points:
        x, y = point
        cv2.circle(anotated_mask, (x, y), 5, (0, 0, 255), -1)

    return anotated_mask


def get_masks_with_annotations(mask, bboxes, points):
    annotated_mask = cv2.merge([mask, mask, mask])

    if points is None:
        return get_masks_with_bbox(mask, bboxes)

    mask_bbox = get_masks_with_bbox(mask, bboxes)
    annotated_mask = get_masks_with_point(mask_bbox, points)

    return annotated_mask


def logits_to_probabilities(logits: np.ndarray) -> np.ndarray:
    safe_value = np.log(np.finfo(logits.dtype).max)
    clipped_logits = np.clip(logits, -safe_value, safe_value)
    probabilities = 1 / (1 + np.exp(-clipped_logits))

    return probabilities


def tensor_to_np(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a PyTorch tensor to a NumPy array suitable for visualization.

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


def visualize(images) -> None:
    """
    Visualizes a list of images using Matplotlib.

    Images can be provided as NumPy arrays or PyTorch tensors.
    PyTorch tensors will be converted to NumPy arrays automatically.

    Args:
        images (List[Union[np.ndarray, torch.Tensor]]): A list of images where each image
            is either a NumPy array or a PyTorch tensor. Tensors should have shape [C, H, W],
            while NumPy arrays should have shape [H, W, C] or [H, W].
    """
    num_imgs = len(images)
    images_copy = images.copy()

    plt.figure(figsize=(10, 5))
    for i in range(num_imgs):
        if isinstance(images_copy[i], torch.Tensor):
            images_copy[i] = tensor_to_np(images_copy[i])
        cmap = None
        if images_copy[i].shape[-1] == 1 or len(images_copy[i].shape) == 2:
            cmap = "gray"
        plt.subplot(1, num_imgs, i + 1)
        plt.imshow(images_copy[i], cmap=cmap)
        plt.axis("off")
    plt.show()