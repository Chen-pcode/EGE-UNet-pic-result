import csv
import os

import numpy as np


def _as_bool(mask, threshold=0.5):
    return np.asarray(mask) >= threshold


def _surface(mask):
    try:
        from scipy import ndimage
    except Exception:
        return None

    mask = _as_bool(mask)
    if not mask.any():
        return mask
    structure = ndimage.generate_binary_structure(mask.ndim, 1)
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    return mask ^ eroded


def _surface_distances(pred, target):
    try:
        from scipy import ndimage
    except Exception:
        return None

    pred = _as_bool(pred)
    target = _as_bool(target)
    if not pred.any() and not target.any():
        return np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32)
    if not pred.any() or not target.any():
        return np.array([np.nan], dtype=np.float32), np.array([np.nan], dtype=np.float32)

    pred_surface = _surface(pred)
    target_surface = _surface(target)
    pred_to_target = ndimage.distance_transform_edt(~target_surface)[pred_surface]
    target_to_pred = ndimage.distance_transform_edt(~pred_surface)[target_surface]
    return pred_to_target, target_to_pred


def boundary_f1_score(pred, target, tolerance=2):
    try:
        from scipy import ndimage
    except Exception:
        return np.nan

    pred_surface = _surface(pred)
    target_surface = _surface(target)
    if pred_surface is None or target_surface is None:
        return np.nan
    if not pred_surface.any() and not target_surface.any():
        return 1.0
    if not pred_surface.any() or not target_surface.any():
        return 0.0

    structure = ndimage.generate_binary_structure(pred_surface.ndim, 1)
    pred_dilated = ndimage.binary_dilation(pred_surface, structure=structure, iterations=tolerance)
    target_dilated = ndimage.binary_dilation(target_surface, structure=structure, iterations=tolerance)

    precision = (pred_surface & target_dilated).sum() / max(pred_surface.sum(), 1)
    recall = (target_surface & pred_dilated).sum() / max(target_surface.sum(), 1)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def segmentation_metrics(preds, targets, threshold=0.5, boundary_tolerance=2):
    preds = _as_bool(preds, threshold)
    targets = _as_bool(targets, 0.5)

    tp = np.logical_and(preds, targets).sum()
    fp = np.logical_and(preds, ~targets).sum()
    fn = np.logical_and(~preds, targets).sum()

    miou = float(tp / (tp + fp + fn)) if (tp + fp + fn) else 1.0
    dsc = float((2 * tp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 1.0

    hd95_values = []
    assd_values = []
    boundary_values = []
    for pred, target in zip(preds, targets):
        distances = _surface_distances(pred, target)
        if distances is not None:
            pred_to_target, target_to_pred = distances
            all_distances = np.concatenate([pred_to_target, target_to_pred])
            if not np.isnan(all_distances).all():
                hd95_values.append(float(np.nanpercentile(all_distances, 95)))
                assd_values.append(float(np.nanmean(all_distances)))
        boundary_values.append(boundary_f1_score(pred, target, tolerance=boundary_tolerance))

    return {
        'mIoU': miou,
        'DSC': dsc,
        'HD95': float(np.nanmean(hd95_values)) if hd95_values else np.nan,
        'ASSD': float(np.nanmean(assd_values)) if assd_values else np.nan,
        'Boundary_F1': float(np.nanmean(boundary_values)) if boundary_values else np.nan,
    }


def model_complexity(model, input_size):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    gflops = np.nan
    try:
        import torch
        from thop import profile

        device = next(model.parameters()).device
        dummy = torch.randn(1, *input_size, device=device)
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        gflops = flops / 1e9
    except Exception as exc:
        print('GFLOPs calculation skipped. Install thop on Kaggle with: pip install thop')
        print('Reason:', exc)
    return params / 1e6, gflops


def write_metrics_csv(metrics, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
