import torch
from torch import nn

from depth_router.solver import (
    KLFaultPolicy,
    Move,
    SolverState,
    SwitchableColdMLP,
    coalesce_faults,
    kl_nats,
    unique_fault_bytes,
)


def test_solver_state_makes_kv_write_part_of_the_dynamics() -> None:
    state = SolverState(hidden=torch.zeros(1, 4))
    state = state.after_write(128)
    assert state.kv_written
    assert state.bytes_moved == 128

    try:
        state.after_write(1)
    except RuntimeError:
        pass
    else:
        raise AssertionError("second KV write must be illegal")


def test_faults_are_coalesced_by_predicted_slice() -> None:
    slices = torch.tensor([2, 1, 2, 2, 1, 3])
    plan = coalesce_faults(slices)

    assert sorted(plan) == [1, 2, 3]
    assert plan[2].tolist() == [0, 2, 3]
    assert unique_fault_bytes(slices.tolist(), {1: 10, 2: 20, 3: 30}) == 60


def test_prediction_kl_is_zero_for_same_distribution() -> None:
    p = torch.log_softmax(torch.tensor([[0.0, 1.0, 2.0]]), dim=-1)
    q = torch.log_softmax(torch.tensor([[2.0, 1.0, 0.0]]), dim=-1)

    assert torch.allclose(kl_nats(p, p), torch.zeros(1), atol=1e-7)
    assert float(kl_nats(q, p)) > 0


def test_cold_mlp_batches_all_faulting_tokens_into_one_call() -> None:
    torch.manual_seed(0)
    base = nn.Sequential(
        nn.Linear(4, 8, bias=False),
        nn.SiLU(),
        nn.Linear(8, 4, bias=False),
    )
    cold = SwitchableColdMLP(base)
    x = torch.randn(2, 3, 4)
    mask = torch.tensor([[True, False, True], [False, True, False]])
    cold.set_fault_mask(mask)

    got = cold(x)

    flat = x.reshape(-1, 4)
    idx = torch.nonzero(mask.reshape(-1), as_tuple=False).flatten()
    expected = torch.zeros_like(flat)
    expected.index_copy_(0, idx, base(flat.index_select(0, idx)))

    assert torch.allclose(got, expected.reshape_as(x))
    assert cold.calls == 1
    assert cold.tokens_faulted == 3


def test_kl_policy_faults_first_then_uses_prediction_movement() -> None:
    policy = KLFaultPolicy(0.05)

    assert bool(policy.fault_mask(0, None))
    assert not bool(policy.fault_mask(1, torch.tensor(0.01)))
    assert bool(policy.fault_mask(2, torch.tensor(0.06)))


def test_state_tracks_spin_and_fault_bytes() -> None:
    h = torch.zeros(1, 2)
    p = torch.log_softmax(torch.tensor([[0.0, 1.0]]), dim=-1)
    state = SolverState(hidden=h)
    state = state.after_fault("mlp", 1024)
    state = state.after_spin(h + 1, p, 64)

    assert state.resident == frozenset({"mlp"})
    assert state.faults == 1
    assert state.spins == 1
    assert state.bytes_moved == 1088
    assert state.log_probs is p
