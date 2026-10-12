"""Static guard against evaluating right-padding positions as next tokens."""

from pathlib import Path

AFFECTED_BATCHED_LAST_POSITION_SCRIPTS = [
    "hf_binary_address.py",
    "hf_cascaded_address.py",
    "hf_causal_state_codebook.py",
    "hf_decode_rollout_hot_residual.py",
    "hf_distilled_address.py",
    "hf_end_to_end_sparsity.py",
    "hf_global_lowbit_decode.py",
    "hf_global_sparse_oracle.py",
    "hf_greedy_end_to_end.py",
    "hf_greedy_subset_oracle.py",
    "hf_hot_residual_sparse.py",
    "hf_hybrid_address.py",
    "hf_logit_tangent_oracle.py",
    "hf_loop_delta_scan.py",
    "hf_lowbit_residual_pursuit.py",
    "hf_swiglu_compile.py",
    "hf_tangent_distilled_address.py",
    "hf_vertical_sparsity.py",
]


def test_batched_last_position_harnesses_use_left_padding():
    experiments = Path(__file__).resolve().parents[1] / "experiments"
    for name in AFFECTED_BATCHED_LAST_POSITION_SCRIPTS:
        source = (experiments / name).read_text(encoding="utf-8")
        assert 'tokenizer.padding_side = "left"' in source, name
