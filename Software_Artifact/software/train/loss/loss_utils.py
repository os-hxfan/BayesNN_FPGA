from typing import Tuple
# All of below functions are from https://github.com/mary-phuong/multiexit-distillation/blob/master/main.py
def binary_accuracy(logit, y, apply_sigmoid=True, reduce=True) -> float:
    prob = logit.sigmoid() if apply_sigmoid else logit
    pred = prob.round().long().view(-1)
    return (pred == y).float().mean() if reduce else (pred == y).float()


def multiclass_accuracy(scores, y, reduce=True):
    _, pred = scores.max(dim=1)
    return (pred == y).float().mean() if reduce else (pred == y).float()


def multiclass_accuracies(scores, y, tops: Tuple[int]):
    _, pred = scores.topk(k=max(tops), dim=1)
    labelled = (y != -100)
    if not any(labelled):
        return [1.0 for i in tops]
    hit = (pred[labelled] == y[labelled, None])
    # Added a cpu() call to avoid type error when converting to numpy later
    topk_acc = hit.float().cumsum(dim=1).mean(dim=0).cpu()
    return [topk_acc[i-1] for i in tops]