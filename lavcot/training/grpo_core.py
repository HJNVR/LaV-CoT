"""A small, dependency-light implementation of the core GRPO objective.

This is intentionally separate from TRL so the optimization math is inspectable
and unit-testable. The production entry point uses TRL for distributed
generation, reference-model management, checkpointing, and multimodal batching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn


def compute_group_advantages(
    rewards: Tensor, group_size: int, eps: float = 1e-4
) -> Tensor:
    """Standardize rewards within each prompt's sampled completion group."""
    if rewards.ndim != 1:
        raise ValueError("rewards must be a flat tensor")
    if group_size < 2 or rewards.numel() % group_size:
        raise ValueError("group_size must divide rewards and be at least 2")
    grouped = rewards.float().view(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True, unbiased=False)
    return ((grouped - mean) / (std + eps)).reshape_as(rewards)


def selective_log_softmax(logits: Tensor, token_ids: Tensor) -> Tensor:
    """Return log p(token_ids) without materializing a second vocabulary tensor."""
    return logits.log_softmax(dim=-1).gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)


def per_token_kl(policy_logps: Tensor, reference_logps: Tensor) -> Tensor:
    """TRL/GRPO's non-negative exponential KL estimator."""
    log_ratio = reference_logps - policy_logps
    return log_ratio.exp() - log_ratio - 1.0


@dataclass
class GRPOOutput:
    loss: Tensor
    policy_loss: Tensor
    kl: Tensor
    clip_fraction: Tensor


class GRPOLoss(nn.Module):
    def __init__(self, beta: float = 0.04, epsilon: float = 0.2) -> None:
        super().__init__()
        self.beta = beta
        self.epsilon = epsilon

    def forward(
        self,
        policy_logps: Tensor,
        old_logps: Tensor,
        reference_logps: Tensor,
        advantages: Tensor,
        completion_mask: Tensor,
    ) -> GRPOOutput:
        if policy_logps.shape != completion_mask.shape:
            raise ValueError("log probabilities and completion_mask must match")
        if advantages.ndim == 1:
            advantages = advantages[:, None]
        ratio = (policy_logps - old_logps).exp()
        unclipped = ratio * advantages
        clipped = ratio.clamp(1 - self.epsilon, 1 + self.epsilon) * advantages
        policy = -torch.minimum(unclipped, clipped)
        kl = per_token_kl(policy_logps, reference_logps)
        token_loss = policy + self.beta * kl
        denom = completion_mask.sum(dim=1).clamp_min(1)
        loss = ((token_loss * completion_mask).sum(dim=1) / denom).mean()
        policy_loss = ((policy * completion_mask).sum(dim=1) / denom).mean()
        mean_kl = ((kl * completion_mask).sum(dim=1) / denom).mean()
        clipped_tokens = ((ratio - 1).abs() > self.epsilon).float()
        clip_fraction = (
            (clipped_tokens * completion_mask).sum(dim=1) / denom
        ).mean()
        return GRPOOutput(loss, policy_loss, mean_kl, clip_fraction)
