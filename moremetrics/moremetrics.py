import torch
from kornia.losses import CharbonnierLoss, WelschLoss, CauchyLoss, GemanMcclureLoss
from torchmetrics import Metric

from .xcorr2 import XCorr2


class Loss2Metric(Metric):
    is_differentiable = True

    def __init__(self, LossClass, **kwargs):
        super().__init__()
        self.lossfn = LossClass(**kwargs)
        self.add_state("loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        self.loss += self.lossfn(preds, target).sum()
        self.count += len(preds)

    def compute(self):
        return self.loss.float() / self.count


class ZeroNormalizedCrossCorrelation(Loss2Metric):
    higher_is_better = True

    def __init__(self):
        super().__init__(XCorr2, zero_mean_normalized=True)


class Charbonnier(Loss2Metric):
    def __init__(self, reduction="mean"):
        super().__init__(CharbonnierLoss, reduction=reduction)


class Welsch(Loss2Metric):
    def __init__(self, reduction="mean"):
        super().__init__(WelschLoss, reduction=reduction)


class Cauchy(Loss2Metric):
    def __init__(self, reduction="mean"):
        super().__init__(CauchyLoss, reduction=reduction)


class GemanMcClure(Loss2Metric):
    def __init__(self, reduction="mean"):
        super().__init__(GemanMcclureLoss, reduction=reduction)
