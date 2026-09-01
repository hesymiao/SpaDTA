from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence as kldiv
from torch.autograd import Variable

from .distributions import ZeroInflatedGaussian


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
        log_mu: torch.Tensor = None,
        theta: torch.Tensor = None,
        gate_logits: torch.Tensor = None,
        reduction: str = "sum",
    ) -> torch.Tensor:
        if theta is None:
            theta = total_counts
        if theta is None:
            raise ValueError("theta or total_counts must be provided")
        if log_mu is None:
            if logits is not None:
                log_mu = logits + torch.log(theta)
            elif mu is not None:
                log_mu = torch.log(mu)
            else:
                raise ValueError("mu/log_mu or total_counts/logits must be provided")
        if gate_logits is None:
            raise ValueError("gate_logits must be provided")

        log_theta = torch.log(theta)
        log_theta_plus_mu = torch.logaddexp(log_theta, log_mu)
        nb_positive_term = (
            torch.lgamma(X + theta)
            - torch.lgamma(theta)
            - torch.lgamma(X + 1.0)
            + theta * (log_theta - log_theta_plus_mu)
            + X * (log_mu - log_theta_plus_mu)
        )
        nb_zero_term = theta * (log_theta - log_theta_plus_mu)
        log_gate = F.logsigmoid(gate_logits)
        log_not_gate = F.logsigmoid(-gate_logits)
        nonzero_log_prob = log_not_gate + nb_positive_term
        zero_log_prob = torch.logaddexp(log_gate, log_not_gate + nb_zero_term)
        zinb_log_prob = torch.where(X == 0, zero_log_prob, nonzero_log_prob)
        if reduction == "sum":
            return -zinb_log_prob.sum(dim=1)
        if reduction == "mean":
            return -zinb_log_prob.mean(dim=1)
        if reduction == "none":
            return -zinb_log_prob
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
