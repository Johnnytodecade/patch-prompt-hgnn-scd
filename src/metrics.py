import torch


def confusion_matrix(prediction, target, classes, ignore_index=255):
    valid = target != ignore_index
    ids = target[valid] * classes + prediction[valid]
    return torch.bincount(ids, minlength=classes * classes).reshape(classes, classes)


def scores(matrix):
    matrix = matrix.float()
    tp = matrix.diag()
    precision = tp / matrix.sum(0).clamp_min(1)
    recall = tp / matrix.sum(1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
    iou = tp / (matrix.sum(0) + matrix.sum(1) - tp).clamp_min(1)
    return {"f1": f1.mean().item(), "miou": iou.mean().item(),
            "oa": (tp.sum() / matrix.sum().clamp_min(1)).item()}
