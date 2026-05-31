from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence as kldiv
from torch.autograd import Variable

from .distributions import NegativeBinomial, ZeroInflatedGaussian, ZeroInflatedNegativeBinomial


def pairwise_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.view(x.shape[0], x.shape[1], 1)
    y = torch.transpose(y, 0, 1)
    output = torch.sum((x - y) ** 2, 1)
    return torch.transpose(output, 0, 1)


def gaussian_kernel_matrix(x: torch.Tensor, y: torch.Tensor, alphas: torch.Tensor) -> torch.Tensor:
    dist = pairwise_distance(x, y).contiguous()
    dist_ = dist.view(1, -1)
    alphas = alphas.view(alphas.shape[0], 1)
    beta = 1.0 / (2.0 * alphas)
    s = torch.matmul(beta, dist_)
    return torch.sum(torch.exp(-s), 0).view_as(dist)


def mmd_loss_calc(source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
    alphas = [
        1e-6,
        1e-5,
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        100,
        1e3,
        1e4,
        1e5,
        1e6,
    ]
    alpha_tensor = Variable(torch.FloatTensor(alphas)).to(device=source_features.device)
    cost = torch.mean(gaussian_kernel_matrix(source_features, source_features, alpha_tensor))
    cost += torch.mean(gaussian_kernel_matrix(target_features, target_features, alpha_tensor))
    cost -= 2 * torch.mean(gaussian_kernel_matrix(source_features, target_features, alpha_tensor))
    return cost


class LossFunction:
    @staticmethod
    def mmd_loss_trvae(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return mmd_loss_calc(x, y)

    @staticmethod
    def zinb_reconstruction_loss(
        X: torch.Tensor,
        total_counts: torch.Tensor = None,
        logits: torch.Tensor = None,
        mu: torch.Tensor = None,
        theta: torch.Tensor = None,
        gate_logits: torch.Tensor = None,
        reduction: str = "sum",
    ) -> torch.Tensor:
        if total_counts is None and logits is None:
            if mu is None or theta is None:
                raise ValueError("mu/theta or total_counts/logits must be provided")
            logits = (mu / theta).log()
            total_counts = theta + 1e-6
        znb = ZeroInflatedNegativeBinomial(
            total_count=total_counts,
            logits=logits,
            gate_logits=gate_logits,
        )
        if reduction == "sum":
            return -znb.log_prob(X).sum(dim=1)
        if reduction == "mean":
            return -znb.log_prob(X).mean(dim=1)
        if reduction == "none":
            return -znb.log_prob(X)
        raise ValueError(f"Unsupported reduction: {reduction}")

    @staticmethod
    def zi_gaussian_reconstruction_loss(
        X: torch.Tensor,
        mean: torch.Tensor,
        variance: torch.Tensor,
        gate_logits: torch.Tensor,
        reduction: Literal["sum", "mean"] = "sum",
    ) -> torch.Tensor:
        zg = ZeroInflatedGaussian(mean=mean, variance=variance, gate_logits=gate_logits)
        if reduction == "sum":
            return -zg.log_prob(X).sum(dim=1)
        if reduction == "mean":
            return -zg.log_prob(X).mean(dim=1)
        raise ValueError(f"Unsupported reduction: {reduction}")

    @staticmethod
    def gaussian_reconstruction_loss(
        X: torch.Tensor,
        mean: torch.Tensor,
        variance: torch.Tensor,
        reduction: Literal["sum", "mean", "none"] = "sum",
    ) -> torch.Tensor:
        g = Normal(mean, variance)
        if reduction == "sum":
            return -g.log_prob(X).sum(dim=1)
        if reduction == "mean":
            return -g.log_prob(X).mean(dim=1)
        if reduction == "none":
            return -g.log_prob(X)
        raise ValueError(f"Unsupported reduction: {reduction}")
