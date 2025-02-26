import warnings
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm

from metric import MetricComputer
from utils import (
    create_bbox_prompt,
    create_point_prompt,
    get_masks_with_annotations,
    logits_to_probabilities,
    mask_postprocessing,
    modeset,
    separate_objects,
)

warnings.filterwarnings("ignore", category=FutureWarning)

categories = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


def parse_args():
    parser = ArgumentParser(description="Inference SAM model")

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="results", required=False)
    parser.add_argument("--mode", type=str, default="naive",required=False, help="select mode in: ['naive', 'b', 'bp', 'bps']")
    parser.add_argument("--model_type", type=str, default="vit_h", required=False)
    parser.add_argument("--device", type=str, default="cuda:0", required=False)
    parser.add_argument("--n_points", type=int, default=20, required=False)
    parser.add_argument("--img_preproc", type=bool, default=True, required=False)

    args = parser.parse_args()
    print(args)

    return args


def test(category: str, data_dir: str, save_dir: str, metric_save_dir: str, args):
    mode = args.mode
    n_points = args.n_points
    img_preprocessing = args.img_preproc
    device = args.device
    model_type = args.model_type
    
    if model_type == "vit_h":
        sam_checkpoint = "weights/sam_vit_h_4b8939.pth"

    img_paths = [p for p in data_dir.glob("test/*/*.png") if "good" not in str(p)]
    mask_paths = [
        str(data_dir / "ground_truth" / p.parent.name / p.stem) + "_mask.png"
        for p in img_paths
    ]

    iou = []
    p_auroc = []
    metric_df = pd.DataFrame()
    metric = MetricComputer()

    mode_bbox, mode_point = modeset(mode, n_points)

    for img_p, mask_p in tqdm(zip(img_paths, mask_paths)):
        img_color = cv2.imread(str(img_p))
        
        if img_preprocessing:
            img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
            img = cv2.equalizeHist(img)
            img = np.stack([img] * 3, axis=-1)

        mask = cv2.imread(str(mask_p), 0)
        masks = separate_objects(mask)

        # Create prompt inputs of SAM (bbox, point)
        bbox_prompt = create_bbox_prompt(masks, mode_bbox)
        point_prompt, labels = create_point_prompt(masks, mode_point, n_points)
        annotated_mask = get_masks_with_annotations(mask, bbox_prompt, point_prompt)

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        predictor.set_image(img)
        # mask_input = np.expand_dims(mask, axis=0)

        # Case 1: Only a single Anomalous region
        if len(bbox_prompt) == 1:
            sam_masks_logits, _, _ = predictor.predict(
                point_coords=point_prompt,
                point_labels=labels,
                box=bbox_prompt,
                # mask_input=mask_input,
                multimask_output=False,
                return_logits=True,
            )
            sam_mask_binary = sam_masks_logits[0] > predictor.model.mask_threshold
            sam_mask_binary = np.where(sam_mask_binary == True, 1, 0)
            sam_mask_binary = mask_postprocessing(sam_mask_binary)  # (H, W)

            # metric
            gt_mask = np.expand_dims(mask, axis=0)  # (1, H, W)
            gt_mask_binary = np.where(gt_mask >= 128, 1, 0)
            sam_mask_binary_exp = np.expand_dims(sam_mask_binary, axis=0)  # (1, H, W)

            sam_mask_prob = logits_to_probabilities(sam_masks_logits)
            sam_mask_prob_t = torch.from_numpy(sam_mask_prob)  # (1, H, W)
            gt_mask_t = torch.from_numpy(gt_mask_binary)  # (1, H, W)
            sam_masks_binary_t = torch.from_numpy(sam_mask_binary_exp)  # (1, H, W)

            iou.append(metric.compute_iou(sam_masks_binary_t, gt_mask_t, threshold=0))
            p_auroc.append(metric.compute_p_auroc(sam_mask_prob_t, gt_mask_t))

        # Case 2: Multiple Anomalous regions
        elif len(bbox_prompt) > 1:
            bbox_prompt = torch.tensor(bbox_prompt, device=device) # (n_mask, 4)
            point_prompt = torch.tensor(point_prompt, device=device) # (n_mask, n_point, 2)
            labels = torch.tensor(labels, device=device) # (n_mask, n_point)

            transformed_boxes = predictor.transform.apply_boxes_torch(
                bbox_prompt, img.shape[:2]
            )
            transformed_points = predictor.transform.apply_coords_torch(
                point_prompt, img.shape[:2]
            )

            sam_masks_logits, _, _ = predictor.predict_torch( # (n_mask, 1, H, W)
                point_coords=transformed_points,
                point_labels=labels,
                boxes=transformed_boxes,
                multimask_output=False,
                return_logits=True,
            )
            sam_masks_logits = sam_masks_logits.detach().cpu().numpy()
            sam_mask_binary = np.zeros(sam_masks_logits.shape[-2:])
            merged_logits = np.zeros(sam_masks_logits.shape[-2:])

            for i, m in enumerate(sam_masks_logits):
                m = m > predictor.model.mask_threshold
                m = np.where(m[0] == True, 1, 0)
                m = mask_postprocessing(m)
                sam_mask_binary += m
                filtered_logit = sam_masks_logits[i][0] * m
                merged_logits += filtered_logit

            merged_logits = merged_logits[np.newaxis, ...]
            logits_prob = logits_to_probabilities(merged_logits)
            gt_mask = np.expand_dims(mask, axis=0)
            gt_mask_binary = np.where(gt_mask >= 128, 1, 0)
            sam_mask_binary_exp = np.expand_dims(sam_mask_binary, axis=0)

            logits_prob = torch.from_numpy(logits_prob)  # (1, H, W)
            gt_mask_t = torch.from_numpy(gt_mask_binary)  # (1, H, W)
            sam_masks_binary_t = torch.from_numpy(sam_mask_binary_exp)  # (1, H, W)

            iou.append(metric.compute_iou(sam_masks_binary_t, gt_mask_t, threshold=0))
            p_auroc.append(metric.compute_p_auroc(logits_prob, gt_mask_t))

        if len(iou) == len(img_paths):
            mean_iou = sum(iou) / len(iou)
            mean_p_auroc = sum(p_auroc) / len(p_auroc)
            mean_row = pd.DataFrame(
                {
                    "class": category,
                    "iou": [round(mean_iou, 4)],
                    "p_auroc": [round(mean_p_auroc, 4)],
                }
            )
            metric_df = pd.concat([metric_df, mean_row], ignore_index=True)
            metric_df.to_csv(metric_save_dir, index=False)

        save_path = save_dir / img_p.parent.name
        save_path.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(save_path / img_p.name), img_color)
        cv2.imwrite(str(save_path / img_p.stem) + "_gt.png", mask)
        cv2.imwrite(str(save_path / img_p.stem) + "_gt_prompt.png", annotated_mask)
        cv2.imwrite(str(save_path / img_p.stem) + "_sam.png", sam_mask_binary)


def main():
    args = parse_args()
    for category in categories:
        print("=" * 30)
        print(f"Testing for Category: {category}")
        print("=" * 30)

        data_dir = Path(args.data_dir) / category
        save_dir = Path(args.save_dir) / category
        metric_save_dir = Path(save_dir) / "metric.csv"

        test(category, data_dir, save_dir, metric_save_dir, args)


if __name__ == "__main__":
    main()