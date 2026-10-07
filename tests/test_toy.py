import torch

from depth_router.toy import PointerChaseSpec, pointer_chase_batch


def test_pointer_chase_batch_shapes_and_ranges() -> None:
    spec = PointerChaseSpec(nodes=8, max_hops=4)
    x, y, hops = pointer_chase_batch(16, spec)

    assert x.shape == (16, spec.seq_len)
    assert y.shape == (16,)
    assert hops.shape == (16,)
    assert int(x.min()) >= 0
    assert int(x.max()) < spec.vocab_size
    assert int(y.min()) >= 0
    assert int(y.max()) < spec.nodes
    assert int(hops.min()) >= 1
    assert int(hops.max()) <= spec.max_hops


def test_pointer_chase_is_reproducible_with_generator() -> None:
    spec = PointerChaseSpec(nodes=5, max_hops=3)
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)

    batch1 = pointer_chase_batch(4, spec, generator=g1)
    batch2 = pointer_chase_batch(4, spec, generator=g2)

    for left, right in zip(batch1, batch2, strict=True):
        assert torch.equal(left, right)
