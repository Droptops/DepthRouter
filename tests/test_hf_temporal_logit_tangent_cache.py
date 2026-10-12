"""Claim-boundary guards for the temporal logit-tangent oracle."""

from pathlib import Path


def test_temporal_tangent_oracle_does_not_label_logical_misses_physical():
    experiments = Path(__file__).resolve().parents[1] / "experiments"
    source = (experiments / "hf_temporal_logit_tangent_cache.py").read_text(
        encoding="utf-8"
    )
    assert '"steady_state_logical_neuron_miss_fraction"' in source
    assert '"bytes and latency remain unknown until confirmed' in source
    assert "norm.cpu()" not in source
    assert "lm_head.cpu()" not in source

    for name in [
        "hf_temporal_logit_tangent_cache.py",
        "hf_temporal_tangent_distilled.py",
        "hf_temporal_address_cache.py",
    ]:
        source = (experiments / name).read_text(encoding="utf-8")
        assert "physical_fault" not in source, name
