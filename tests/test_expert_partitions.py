import pytest
import torch
from torch import nn

from depth_router.expert_partitions import (
    AssignmentRecipe,
    balanced_round,
    candidate_combinations,
    coactivation_partition,
    fit_assignment_pair,
    partition_indices,
    random_partition,
    sinkhorn_assignment,
    straight_through_assignment,
    validate_partition,
)


def test_qwen_partition_integrity_and_exact_candidate_enumeration() -> None:
    labels = random_partition(4864, seed=17)
    validate_partition(labels)
    assert torch.bincount(labels).tolist() == [608] * 8
    assert torch.equal(labels, random_partition(4864, seed=17))
    assert not torch.equal(labels, random_partition(4864, seed=29))
    combos = candidate_combinations()
    assert combos.shape == (28, 6)
    assert len({tuple(row) for row in combos.tolist()}) == 28
    assert torch.bincount(combos.flatten()).tolist() == [21] * 8
    for combo in combos:
        ids = partition_indices(labels, combo)
        assert len(ids) == ids.unique().numel() == 3648
        assert set(labels[ids].tolist()) == set(combo.tolist())


def test_mathematical_permutation_indexes_gate_up_and_down_together() -> None:
    torch.manual_seed(3)
    gate, up = nn.Linear(5, 16).double(), nn.Linear(5, 16).double()
    down = nn.Linear(16, 5).double()
    x = torch.randn(7, 5, dtype=torch.float64)
    expected = down(torch.nn.functional.silu(gate(x)) * up(x))
    permutation = partition_indices(random_partition(16, experts=4, seed=17), experts=4)
    assert torch.equal(permutation.sort().values, torch.arange(16))
    actual = torch.nn.functional.linear(
        torch.nn.functional.silu(
            torch.nn.functional.linear(x, gate.weight[permutation], gate.bias[permutation])
        )
        * torch.nn.functional.linear(x, up.weight[permutation], up.bias[permutation]),
        down.weight[:, permutation],
        down.bias,
    )
    # This tests real-arithmetic wiring in float64; BF16 controls are measured
    # separately in the complete-model experiment, never inferred from here.
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_invalid_partitions_indices_and_dimensions_are_rejected() -> None:
    with pytest.raises(ValueError, match="divisible"):
        random_partition(15, experts=4, seed=1)
    with pytest.raises(ValueError, match="balanced"):
        validate_partition(torch.tensor([0, 0, 0, 1]), experts=2)
    with pytest.raises(ValueError, match="range"):
        validate_partition(torch.tensor([0, 1, 2, 3]), experts=2)
    with pytest.raises(ValueError, match="torch.long"):
        validate_partition(torch.tensor([0.0, 0.0, 1.0, 1.0]), experts=2)
    with pytest.raises(ValueError, match="unique"):
        partition_indices(torch.tensor([0, 0, 1, 1]), [0, 0], experts=2)
    with pytest.raises(ValueError, match="integer"):
        partition_indices(torch.tensor([0, 0, 1, 1]), [0.5], experts=2)
    with pytest.raises(ValueError, match="active"):
        candidate_combinations(experts=8, active=9)


def test_global_greedy_projection_has_deterministic_ties_and_capacities() -> None:
    scores = torch.zeros(8, 4)
    assert balanced_round(scores).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    # Every neuron prefers expert 0; hard capacity still holds exactly.
    scores[:, 0] = 100
    validate_partition(balanced_round(scores), experts=4)
    scores[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        balanced_round(scores)


def test_sinkhorn_marginals_and_straight_through_gradient() -> None:
    torch.manual_seed(9)
    logits = torch.randn(24, 4, requires_grad=True)
    soft = sinkhorn_assignment(logits, temperature=0.7, iterations=50)
    torch.testing.assert_close(soft.sum(dim=1), torch.ones(24), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(soft.sum(dim=0), torch.full((4,), 6.0), atol=1e-6, rtol=1e-6)
    hard, labels = straight_through_assignment(logits, temperature=0.7, iterations=50)
    assert torch.equal(hard, torch.nn.functional.one_hot(labels, 4).float())
    probe = torch.randn_like(hard)
    hard_gradient = torch.autograd.grad((hard * probe).sum(), logits)[0]
    soft_gradient = torch.autograd.grad((soft * probe).sum(), logits)[0]
    assert bool(torch.isfinite(hard_gradient).all())
    assert hard_gradient.abs().sum() > 0
    torch.testing.assert_close(hard_gradient, soft_gradient, atol=0, rtol=0)


def test_coactivation_clustering_is_finite_balanced_and_seeded() -> None:
    generator = torch.Generator().manual_seed(18)
    activations = torch.randn(20, 32, generator=generator)
    a = coactivation_partition(activations, experts=4, seed=17, sketch_dim=8, iterations=3)
    b = coactivation_partition(activations, experts=4, seed=17, sketch_dim=8, iterations=3)
    validate_partition(a, experts=4)
    assert torch.equal(a, b)
    # Silent neurons must not introduce NaNs or break exact balance.
    zero = coactivation_partition(torch.zeros_like(activations), experts=4, seed=17, iterations=2)
    validate_partition(zero, experts=4)


def test_pair_matches_initialization_minibatches_candidates_and_selection() -> None:
    torch.manual_seed(8)
    examples = torch.randn(10, 16)
    seen = {"hidden": [], "output": []}

    def identical_objectives(objective, row_ids, masks):
        assert row_ids.device.type == "cpu"
        assert row_ids.max() < len(examples)
        assert torch.equal(masks, masks.round())
        assert masks.sum(dim=1).tolist() == [12.0, 12.0]
        seen[objective].append((row_ids.clone(), masks.detach().clone()))
        omitted = examples[row_ids][None] * (1 - masks[:, None])
        return omitted.sum(dim=-1).square()

    initial = random_partition(16, experts=4, seed=17)
    partitions, audit = fit_assignment_pair(
        initial,
        len(examples),
        identical_objectives,
        seed=17,
        device="cpu",
        experts=4,
        active=3,
        recipe=AssignmentRecipe(steps=4, batch_size=3, candidates_per_step=2),
    )
    assert torch.equal(partitions["hidden"], partitions["output"])
    assert audit["parameter_count"] == 64
    assert audit["router_trained"] is False
    assert audit["checkpoint_rule"] == "final iterate"
    assert audit["initialization_equal"] and audit["minibatch_order_equal"]
    assert audit["candidate_order_equal"]
    assert audit["objectives"]["hidden"] == audit["objectives"]["output"]
    for (rows_c, masks_c), (rows_d, masks_d) in zip(seen["hidden"], seen["output"], strict=True):
        assert torch.equal(rows_c, rows_d)
        assert torch.equal(masks_c, masks_d)
    assert len(seen["hidden"]) == len(seen["output"]) == 4
    assert max(audit["objectives"]["hidden"]["gradient_norm_by_step"]) > 0


def test_pair_validates_callback_and_does_not_hide_budget_exhaustion() -> None:
    initial = random_partition(8, experts=2, seed=17)
    recipe = AssignmentRecipe(steps=1, batch_size=2, candidates_per_step=2)

    def wrong_shape(objective, row_ids, masks):
        return masks.sum().reshape(1)

    with pytest.raises(ValueError, match="loss matrix"):
        fit_assignment_pair(
            initial, 4, wrong_shape, seed=17, device="cpu", experts=2, active=1, recipe=recipe
        )

    def budget_exhausted(objective, row_ids, masks):
        raise TimeoutError("research budget exhausted")

    with pytest.raises(TimeoutError, match="budget exhausted") as captured:
        fit_assignment_pair(
            initial, 4, budget_exhausted, seed=17, device="cpu", experts=2, active=1, recipe=recipe
        )
    partial = captured.value.assignment_partial
    assert partial["audit"]["status"] == "incomplete"
    assert partial["audit"]["current_completed_steps"] == 0
    assert partial["audit"]["current_objective"] == "hidden"
    assert partial["completed_partitions"] == {}
    assert torch.equal(partial["current_partition"], initial)


def test_interrupted_output_fit_preserves_completed_hidden_fit_and_partial_trace() -> None:
    calls = {"hidden": 0, "output": 0}

    def callback(objective, row_ids, masks):
        calls[objective] += 1
        if objective == "output" and calls[objective] == 2:
            raise TimeoutError("research budget exhausted")
        return (masks @ torch.arange(8).float()).square()[:, None].expand(-1, len(row_ids))

    with pytest.raises(TimeoutError) as captured:
        fit_assignment_pair(
            random_partition(8, experts=2, seed=17),
            4,
            callback,
            seed=17,
            device="cpu",
            experts=2,
            active=1,
            recipe=AssignmentRecipe(steps=2, batch_size=2, candidates_per_step=2),
        )
    partial = captured.value.assignment_partial
    assert set(partial["completed_partitions"]) == {"hidden"}
    validate_partition(partial["completed_partitions"]["hidden"], experts=2)
    validate_partition(partial["current_partition"], experts=2)
    assert partial["audit"]["current_completed_steps"] == 1
    assert partial["audit"]["current_objective"] == "output"
    assert len(partial["audit"]["current_loss_by_step"]) == 1
    assert len(partial["audit"]["completed_objectives"]["hidden"]["loss_by_step"]) == 2
