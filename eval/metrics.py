"""Spatial metrics. All inputs are normalised to [0,1] of the original image."""
import numpy as np


def _arr(p):
    return np.asarray(p, dtype=float).reshape(-1, 2)


def resample(traj, n=20):
    """Resample a polyline to n points equally spaced along its arc length."""
    t = _arr(traj)
    if len(t) == 1:
        return np.repeat(t, n, axis=0)
    seg = np.linalg.norm(np.diff(t, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] == 0:
        return np.repeat(t[:1], n, axis=0)
    q = np.linspace(0, s[-1], n)
    return np.stack([np.interp(q, s, t[:, 0]), np.interp(q, s, t[:, 1])], axis=1)


def discrete_frechet(p, q):
    p, q = _arr(p), _arr(q)
    d = np.linalg.norm(p[:, None] - q[None], axis=2)
    ca = np.full(d.shape, np.inf)
    for i in range(len(p)):
        for j in range(len(q)):
            prev = 0.0 if i == 0 and j == 0 else min(
                ca[i - 1, j] if i > 0 else np.inf,
                ca[i, j - 1] if j > 0 else np.inf,
                ca[i - 1, j - 1] if i > 0 and j > 0 else np.inf)
            ca[i, j] = max(prev, d[i, j])
    return float(ca[-1, -1])


def hausdorff(p, q):
    p, q = _arr(p), _arr(q)
    d = np.linalg.norm(p[:, None] - q[None], axis=2)
    return float(max(d.min(1).max(), d.min(0).max()))


def traj_metrics(pred, gt, n=20):
    """DFD/HD on raw polylines; RMSE/ADE on arc-length resampled polylines; FDE on endpoints."""
    p, g = _arr(pred), _arr(gt)
    pr, gr = resample(p, n), resample(g, n)
    err = np.linalg.norm(pr - gr, axis=1)
    k = min(len(p), len(g))  # ShareRobot-Bench official RMSE: index-matched, truncated
    err_idx = np.linalg.norm(p[:k] - g[:k], axis=1)
    return {
        "dfd": discrete_frechet(p, g),
        "hd": hausdorff(p, g),
        "rmse_idx": float(np.sqrt((err_idx ** 2).mean())),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "ade": float(err.mean()),
        "fde": float(np.linalg.norm(p[-1] - g[-1])),
    }


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / union) if union > 0 else 0.0


def point_in_mask_or_box(pt, box):
    x, y = pt
    return float(box[0] <= x <= box[2] and box[1] <= y <= box[3])
