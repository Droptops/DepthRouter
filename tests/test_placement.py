import torch

from depth_router.placement import (
    Move,
    SolverState,
    coalesce_faults,
    kl_nats,
    nats_per_byte,
    schedule_efficiency,
)


def test_solver_state_is_memory_explicit() -> None:
    state = SolverState(
        h=torch.zeros(2, 1, 4),
        pc=torch.zeros(2, dtype=torch.long),
    )
    state.record(Move.WRITE)
    state.resident_slices.add(3)
    state.record(Move.FAULT)
    state.record(Move.SPIN)
    state.record(Move.HALT)

    assert state.resident_slices == {3}
    assert state.trace == [Move.WRITE, Move.FAULT, Move.SPIN, Move.HALT]


def test_kl_nats_zero_for_same_distribution() -> None:
    logp = torch.log_softmax(torch.tensor([[1.0, 2.0, 3.0]]), dim=-1)
    assert torch.allclose(kl_nats(logp, logp), torch.zeros(1), atol=1e-7)


def test_faults_are_coalesced_by_predicted_slice() -> None:
    slices = torch.tensor([0, 1, 0, 1, 1, 2])
    request = torch.tensor([1, 1, 1, 0, 1, 0], dtype=torch.bool)

    groups = coalesce_faults(slices, request)

    assert [(sid, idx.tolist()) for sid, idx in groups] == [
        (0, [0, 2]),
        (1, [1, 4]),
    ]


def test_value_density_and_schedule_efficiency() -> None:
    assert torch.allclose(nats_per_byte(torch.tensor([2.0]), 4), torch.tensor([0.5]))
    assert schedule_efficiency(2.0, 1.5, 100.0) == 0.005
