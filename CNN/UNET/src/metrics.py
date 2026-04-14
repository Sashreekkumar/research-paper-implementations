import numpy as np
from scipy.spatial.distance import directed_hausdorff

def dice_score(pred, target, smooth=1e-6):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def iou_score(pred, target, smooth=1e-6):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    return (intersection + smooth) / (union + smooth)

def rand_error(pred, target):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    tp = (pred * target).sum()
    tn = ((1 - pred) * (1 - target)).sum()

    total = pred.numel()

    return 1 - ((tp + tn) / total)

def hausdorff_error(pred, target):
    """
    Measures worst-case boundary distance between prediction and ground truth.
    """
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()

    pred_pts = np.argwhere(pred)
    target_pts = np.argwhere(target)

    # handle edge cases (empty masks)
    if len(pred_pts) == 0 or len(target_pts) == 0:
        return float("inf")

    d1 = directed_hausdorff(pred_pts, target_pts)[0]
    d2 = directed_hausdorff(target_pts, pred_pts)[0]

    return max(d1, d2)