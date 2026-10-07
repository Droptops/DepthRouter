import torch

from depth_router import DepthRouterConfig, DepthRouterModel, StackedTransformerBaseline
from depth_router.cost import parameter_count


def config() -> DepthRouterConfig:
    return DepthRouterConfig(
        vocab_size=32,
        max_seq_len=10,
        num_classes=8,
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        max_steps=4,
        adapter_rank=4,
    )


def test_recurrent_forward_shape() -> None:
    model = DepthRouterModel(config())
    x = torch.randint(0, 32, (3, 10))
    logits = model(x)
    assert logits.shape == (3, 8)


def test_adaptive_router_can_halt_after_minimum_depth() -> None:
    model = DepthRouterModel(config())
    x = torch.randint(0, 32, (5, 10))
    _, stats = model(
        x,
        adaptive=True,
        halt_threshold=1e9,
        min_steps=1,
        return_stats=True,
    )
    assert stats.sample_passes.tolist() == [1, 1, 1, 1, 1]
    assert stats.physical_block_evaluations == 1


def test_shared_recurrence_uses_fewer_parameters_than_unshared_stack() -> None:
    recurrent = DepthRouterModel(config())
    stacked = StackedTransformerBaseline(config())
    assert parameter_count(recurrent) < parameter_count(stacked)


def test_pass_adapters_are_still_smaller_than_full_unshared_depth() -> None:
    recurrent = DepthRouterModel(config(), use_pass_adapters=True)
    stacked = StackedTransformerBaseline(config())
    assert parameter_count(recurrent) < parameter_count(stacked)
