import torch

from lavcot.training.grpo_core import GRPOLoss, compute_group_advantages


def test_group_advantages_are_normalized_per_prompt():
    advantages = compute_group_advantages(torch.tensor([1.0, 2.0, 4.0, 4.0]), 2)
    groups = advantages.view(2, 2)
    assert torch.allclose(groups.mean(1), torch.zeros(2), atol=1e-5)
    assert torch.allclose(groups[1], torch.zeros(2), atol=1e-5)


def test_grpo_loss_is_finite_and_backpropagates():
    policy = torch.zeros(2, 3, requires_grad=True)
    old = torch.zeros_like(policy)
    reference = torch.zeros_like(policy)
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.float32)
    output = GRPOLoss()(policy, old, reference, torch.tensor([1.0, -1.0]), mask)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert policy.grad is not None
