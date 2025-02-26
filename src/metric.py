from typing import Optional

import numpy as np
import torch
from sklearn.metrics import roc_curve, auc


class MetricComputer:
    """
    A class to compute evaluation metrics for binary classification tasks 
    such as segmentation, including F1-score and IoU. Supports both batch-level 
    and sample-level threshold optimization.
    """
    def compute_f1_score(
        self, pred: torch.Tensor, mask: torch.Tensor, threshold: float
    ) -> float:
        """
        Compute F1-score between predicted and ground truth masks.
        Args:
            pred: torch.Tensor (C, H, W)
            mask: torch.Tensor (C, H, W)
            threshold: float - Threshold to binarize predictions.
        Returns:
            float: F1-score.
        """
        pred_binary = (pred > threshold).float()
        mask_binary = mask.float()

        tp = (pred_binary * mask_binary).sum().item()  # True Positives
        fp = (pred_binary * (1 - mask_binary)).sum().item()  # False Positives
        fn = ((1 - pred_binary) * mask_binary).sum().item()  # False Negatives

        precision = tp / (tp + fp + 1e-8) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn + 1e-8) if tp + fn > 0 else 0.0

        f1_score = (
            (2 * precision * recall) / (precision + recall + 1e-8)
            if precision + recall > 0
            else 0.0
        )
        return f1_score
    
    def compute_p_auroc(
        self, pred: torch.Tensor, mask: torch.Tensor
    ) -> float:
        """
        Compute Pixel-wise Area Under ROC Curve (P-AUROC).
        Args:
            pred: torch.Tensor (C, H, W) - Model output probabilities
            mask: torch.Tensor (C, H, W) - Ground truth binary mask
        Returns:
            float: P-AUROC score.
        """
        pred_flat = pred.view(-1).cpu().numpy()
        mask_flat = mask.view(-1).cpu().numpy()

        fpr, tpr, _ = roc_curve(mask_flat, pred_flat)
        p_auroc = auc(fpr, tpr)

        return p_auroc
    
    def compute_iou(
        self, pred: torch.Tensor, mask: torch.Tensor, threshold: float
    ) -> float:
        """
        Compute IoU between predicted and ground truth masks.
        Args:
            pred: torch.Tensor (C, H, W)
            mask: torch.Tensor (C, H, W)
            threshold: float - Threshold to binarize predictions.
        Returns:
            float: IoU.
        """
        pred_binary = (pred > threshold).float()
        mask_binary = mask.float()

        intersection = (pred_binary * mask_binary).sum()
        union = pred_binary.sum() + mask_binary.sum() - intersection

        iou = (intersection / union).item() if union > 0 else 0.0
        return iou

    def find_optimal_threshold(
        self,
        pred: torch.Tensor,
        mask: torch.Tensor,
        metric_fn: Optional[callable] = None,
        num_thresholds: int = 100,
    ) -> float:
        """
        Find the optimal threshold for a single sample using a given metric function.
        Args:
            pred: torch.Tensor (C, H, W) - Predicted values.
            mask: torch.Tensor (C, H, W) - Ground truth mask.
            metric_fn: callable - Metric function to optimize (e.g., IoU).
            num_thresholds: int - Number of thresholds to test.
        Returns:
            float: Optimal threshold.
        """
        if metric_fn is None:
            metric_fn = self.compute_iou
            
        if torch.unique(pred).numel() == 2:  # If already binary
            return 0.5

        thresholds = torch.linspace(
            pred.min(), pred.max(), steps=num_thresholds, device=pred.device
        )
        best_threshold = 0
        best_metric = 0

        for threshold in thresholds:
            metric = metric_fn(pred, mask, threshold=threshold.item())
            if metric > best_metric:
                best_metric = metric
                best_threshold = threshold.item()

        return best_threshold

    def find_batch_optimal_threshold(
        self,
        preds: torch.Tensor,
        masks: torch.Tensor,
        metric_fn: Optional[callable] = None,
        num_thresholds: int = 100,
    ) -> float:
        """
        Find the optimal threshold for an entire batch using a given metric function.
        Args:
            preds: torch.Tensor (B, C, H, W) - Predicted values for a batch.
            masks: torch.Tensor (B, C, H, W) - Ground truth masks for a batch.
            metric_fn: callable - Metric function to optimize (e.g., IoU).
            num_thresholds: int - Number of thresholds to test.
        Returns:
            float: Optimal threshold for the batch.
        """
        preds_flat = preds.view(-1)
        masks_flat = masks.view(-1)

        if metric_fn is None:
            metric_fn = self.compute_iou
            
        if torch.unique(preds_flat).numel() == 2:  # If already binary
            return 0.5

        thresholds = torch.linspace(
            preds_flat.min(),
            preds_flat.max(),
            steps=num_thresholds,
            device=preds.device,
        )
        best_threshold = 0
        best_metric = 0

        for threshold in thresholds:
            metric = metric_fn(preds_flat, masks_flat, threshold=threshold.item())
            if metric > best_metric:
                best_metric = metric
                best_threshold = threshold.item()

        return best_threshold

    def compute_sample_metrics(
        self, pred: torch.Tensor, mask: torch.Tensor, threshold: float
    ) -> dict:
        """
        Compute F1 and IoU for a single sample with a given threshold.
        Args:
            pred: torch.Tensor (C, H, W) - Predicted values.
            mask: torch.Tensor (C, H, W) - Ground truth mask.
            threshold: float - Threshold to binarize predictions.
        Returns:
            dict: F1, IoU, and the threshold used for this sample.
        """
        f1 = self.compute_f1_score(pred, mask, threshold=threshold)
        iou = self.compute_iou(pred, mask, threshold=threshold)
        p_auroc = self.compute_p_auroc(pred, mask)

        return {"f1": f1, "iou": iou, 'p_auroc': p_auroc}

    def compute_metrics(
        self,
        preds: torch.Tensor,
        masks: torch.Tensor,
        mode: str = "batch",
        num_thresholds: int = 100,
        log_sample: bool = False
    ) -> dict:
        """
        Compute F1 and IoU metrics for a batch of predictions and masks.
        Args:
            preds: torch.Tensor (B, C, H, W) - Predicted values.
            masks: torch.Tensor (B, C, H, W) - Ground truth masks.
            mode: str - "batch" for shared batch-level threshold or "sample" for per-sample threshold.
            num_thresholds: int - Number of thresholds to test for optimal threshold calculation.
            log_sample: bool - Whether to return per-sample metrics.
        Returns:
            dict: Batch-level metrics (and sample-level metrics if log_sample=True).
        """
        f1_scores = []
        iou_scores = []
        p_auroc_scores = []
        sample_results = []
        # Determine threshold based on mode

        if mode == "batch":
            # Batch-level shared threshold
            optimal_threshold = self.find_batch_optimal_threshold(
                preds, masks, num_thresholds=num_thresholds
            )
            # Same threshold for all samples
            thresholds = [optimal_threshold] * preds.size(0)  
        elif mode == "sample":
            # Per-sample thresholds
            thresholds = [
                self.find_optimal_threshold(pred, mask, num_thresholds=num_thresholds)
                for pred, mask in zip(preds, masks)
            ]
        else:
            raise ValueError("Invalid mode. Choose 'batch' or 'sample'.")

        # Compute metrics for each sample
        for idx, (pred, mask, threshold) in enumerate(zip(preds, masks, thresholds)):
            metrics = self.compute_sample_metrics(pred, mask, threshold)
            f1_scores.append(metrics["f1"])
            iou_scores.append(metrics["iou"])
            p_auroc_scores.append(metrics['p_auroc'])
            
            if log_sample:
                sample_results.append(
                    {
                        "sample_idx": idx,
                        "threshold": threshold,
                        "f1": metrics["f1"],
                        "iou": metrics["iou"],
                        "p_auroc": metrics['p_auroc']
                    }
                )

        metrics_result = {
            "f1": np.mean(f1_scores),
            "iou": np.mean(iou_scores),
            "p_auroc": np.mean(p_auroc_scores)
        }

        if log_sample:
            metrics_result["sample_metrics"] = sample_results

        return metrics_result