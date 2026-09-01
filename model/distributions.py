from __future__ import annotations

import torch
from torch.distributions import Distribution, NegativeBinomial, Normal, Poisson, constraints
from torch.distributions.utils import broadcast_all, lazy_property, logits_to_probs, probs_to_logits
from torch.nn.functional import softplus


def broadcast_shape(*shapes, **kwargs):
    strict = kwargs.pop("strict", False)
    reversed_shape = []
    for shape in shapes:
        for i, size in enumerate(reversed(shape)):
            if i >= len(reversed_shape):
                reversed_shape.append(size)
            elif reversed_shape[i] == 1 and not strict:
                reversed_shape[i] = size
            elif reversed_shape[i] != size and (size != 1 or strict):
                raise ValueError(
                    "shape mismatch: objects cannot be broadcast to a single shape: {}".format(
                        " vs ".join(map(str, shapes))
                    )
                )
    return tuple(reversed(reversed_shape))


class ZeroInflatedDistribution(Distribution):
    arg_constraints = {
        "gate": constraints.unit_interval,
        "gate_logits": constraints.real,
    }

    def __init__(self, base_dist, *, gate=None, gate_logits=None, validate_args=None):
        if (gate is None) == (gate_logits is None):
            raise ValueError("Either `gate` or `gate_logits` must be specified, but not both.")
        if gate is not None:
            batch_shape = broadcast_shape(gate.shape, base_dist.batch_shape)
            self.gate = gate.expand(batch_shape)
        else:
            batch_shape = broadcast_shape(gate_logits.shape, base_dist.batch_shape)
            self.gate_logits = gate_logits.expand(batch_shape)
        if base_dist.event_shape:
            raise ValueError(
                "ZeroInflatedDistribution expected empty "
                f"base_dist.event_shape but got {base_dist.event_shape}"
            )
        self.base_dist = base_dist.expand(batch_shape)
        super().__init__(batch_shape, torch.Size(), validate_args)

    @constraints.dependent_property
    def support(self):
        return self.base_dist.support

    @lazy_property
    def gate(self):
        return logits_to_probs(self.gate_logits)

    @lazy_property
    def gate_logits(self):
        return probs_to_logits(self.gate)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        if "gate" in self.__dict__:
            gate, value = broadcast_all(self.gate, value)
            log_prob = (-gate).log1p() + self.base_dist.log_prob(value)
            return torch.where(value == 0, (gate + log_prob.exp()).log(), log_prob)

        gate_logits, value = broadcast_all(self.gate_logits, value)
        base_log_prob = self.base_dist.log_prob(value)
        log_prob_minus_log_gate = -gate_logits + base_log_prob
        log_gate = -softplus(-gate_logits)
        log_prob = log_prob_minus_log_gate + log_gate
        zero_log_prob = softplus(log_prob_minus_log_gate) + log_gate
        return torch.where(value == 0, zero_log_prob, log_prob)

    @lazy_property
    def mean(self):
        return (1 - self.gate) * self.base_dist.mean

    @lazy_property
    def variance(self):
        return (1 - self.gate) * (self.base_dist.mean**2 + self.base_dist.variance) - (self.mean) ** 2


class ZeroInflatedPoisson(ZeroInflatedDistribution):
    arg_constraints = {
        "rate": constraints.positive,
        "gate": constraints.unit_interval,
        "gate_logits": constraints.real,
    }
    support = constraints.nonnegative_integer

    def __init__(self, rate, *, gate=None, gate_logits=None, validate_args=None):
        base_dist = Poisson(rate=rate, validate_args=False)
        base_dist._validate_args = validate_args
        super().__init__(base_dist, gate=gate, gate_logits=gate_logits, validate_args=validate_args)

    @property
    def rate(self):
        return self.base_dist.rate


class ZeroInflatedNegativeBinomial(ZeroInflatedDistribution):
    arg_constraints = {
        "total_count": constraints.greater_than_eq(0),
        "probs": constraints.interval(0.0, 1.0),
        "logits": constraints.real,
        "gate": constraints.unit_interval,
        "gate_logits": constraints.real,
    }
    support = constraints.nonnegative_integer

    def __init__(self, total_count, *, probs=None, logits=None, gate=None, gate_logits=None, validate_args=None):
        base_dist = NegativeBinomial(
            total_count=total_count,
            probs=probs,
            logits=logits,
            validate_args=False,
        )
        base_dist._validate_args = validate_args
        super().__init__(base_dist, gate=gate, gate_logits=gate_logits, validate_args=validate_args)

    @property
    def total_count(self):
        return self.base_dist.total_count

    @property
    def probs(self):
        return self.base_dist.probs

    @property
    def logits(self):
        return self.base_dist.logits


class ZeroInflatedGaussian(ZeroInflatedDistribution):
    arg_constraints = {
        "mean": constraints.real,
        "variance": constraints.greater_than(0),
        "gate": constraints.unit_interval,
        "gate_logits": constraints.real,
    }

    def __init__(self, mean, *, variance, gate=None, gate_logits=None, validate_args=None) -> None:
        base_dist = Normal(mean, variance)
        base_dist._validate_args = validate_args
        super().__init__(base_dist, gate=gate, gate_logits=gate_logits, validate_args=validate_args)

    @property
    def mean(self):
        return self.base_dist.mean

    @property
    def variance(self):
        return self.base_dist.variance
