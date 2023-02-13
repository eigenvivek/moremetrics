import torch


class XCorr2(torch.nn.Module):
    """
    Compute the normalized cross-correlation between two images with the same shape.

    Adapted from https://github.com/connorlee77/pytorch-xcorr2
    """

    def __init__(self, zero_mean_normalized=False):
        super().__init__()
        self.InstanceNorm = torch.nn.InstanceNorm2d(1)
        self.zero_mean_normalized = zero_mean_normalized

    def forward(self, x1, x2):
        assert x1.shape == x2.shape, f"{x1.shape} != {x2.shape}"
        _, c, h, w = x1.shape
        assert c == 1
        if self.zero_mean_normalized:
            x1 = self.InstanceNorm(x1)
            x2 = self.InstanceNorm(x2)
        score = torch.einsum("b...,b...->b", x1, x2)
        score /= h * w * c
        return score
