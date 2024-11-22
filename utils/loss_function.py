import torch as t
import torch.nn as nn
import numpy as np
from skimage.transform import resize


class SaliencyLoss(nn.Module):
    def __init__(self):
        super(SaliencyLoss, self).__init__()

    def forward(self, preds, labels, loss_type="cc"):
        losses = []
        if loss_type == "cc":
            for i in range(labels.shape[0]):  # labels.shape[0] is batch size
                loss = loss_CC(preds[i], labels[i])
                losses.append(loss)

        elif loss_type == "kldiv":
            for i in range(labels.shape[0]):
                loss = loss_KLdiv(preds[i], labels[i])
                losses.append(loss)

        elif loss_type == "sim":
            for i in range(labels.shape[0]):
                loss = loss_similarity(preds[i], labels[i])
                losses.append(loss)

        return t.stack(losses).mean(dim=0, keepdim=True)


# 散度
def loss_KLdiv(pred_map, gt_map):
    eps = 2.2204e-16
    pred_map = pred_map / t.sum(pred_map)
    gt_map = gt_map / t.sum(gt_map)
    div = t.sum(t.mul(gt_map, t.log(eps + t.div(gt_map, pred_map + eps))))
    return div


# 相关性系数
def loss_CC(pred_map, gt_map):
    gt_map_ = gt_map - t.mean(gt_map)
    pred_map_ = pred_map - t.mean(pred_map)
    cc = t.sum(t.mul(gt_map_, pred_map_)) / t.sqrt(
        t.sum(t.mul(gt_map_, gt_map_)) * t.sum(t.mul(pred_map_, pred_map_))
    )
    return cc


# 相似性度量
def loss_similarity(pred_map, gt_map):
    gt_map = (gt_map - t.min(gt_map)) / (t.max(gt_map) - t.min(gt_map))
    gt_map = gt_map / t.sum(gt_map)

    pred_map = (pred_map - t.min(pred_map)) / (t.max(pred_map) - t.min(pred_map))
    pred_map = pred_map / t.sum(pred_map)

    diff = t.min(gt_map, pred_map)
    score = t.sum(diff)

    return score


def normalize(x, method="standard", axis=None):
    # TODO: Prevent divided by zero if the map is flat
    x = np.array(x, copy=False)
    if axis is not None:
        y = np.rollaxis(x, axis).reshape([x.shape[axis], -1])
        shape = np.ones(len(x.shape))
        shape[axis] = x.shape[axis]
        if method == "standard":
            res = (x - np.mean(y, axis=1).reshape(shape)) / np.std(y, axis=1).reshape(
                shape
            )
        elif method == "range":
            res = (x - np.min(y, axis=1).reshape(shape)) / (
                np.max(y, axis=1) - np.min(y, axis=1)
            ).reshape(shape)
        elif method == "sum":
            res = x / np.float_(np.sum(y, axis=1).reshape(shape))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    else:
        if method == "standard":
            res = (x - np.mean(x)) / np.std(x)
        elif method == "range":
            res = (x - np.min(x)) / (np.max(x) - np.min(x))
        elif method == "sum":
            res = x / float(np.sum(x))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    return res
