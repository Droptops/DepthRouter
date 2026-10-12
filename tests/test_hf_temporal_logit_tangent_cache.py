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


def test_tangent_student_harnesses_pin_and_report_revisions():
    experiments = Path(__file__).resolve().parents[1] / "experiments"
    for name in [
        "hf_temporal_tangent_distilled.py",
        "hf_temporal_tangent_exposure_diagnostic.py",
    ]:
        source = (experiments / name).read_text(encoding="utf-8")
        assert 'parser.add_argument("--revision")' in source, name
        assert '"model_revision"' in source, name
        assert '"tokenizer_revision"' in source, name
        assert '"seed": args.seed' in source, name
        assert '"student_amortized_fault_fraction"' not in source, name
        assert '"student_steady_fault_fraction"' not in source, name


def test_tangent_student_separates_teacher_and_execution_widths():
    source = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "hf_temporal_tangent_distilled.py"
    ).read_text(encoding="utf-8")
    assert '"--execution-fraction"' in source
    assert '"teacher_target_fraction": args.fraction' in source
    assert '"selected_payload_fraction": execution_fraction' in source


def test_tangent_student_supports_continuous_rank_distillation():
    source = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "hf_temporal_tangent_distilled.py"
    ).read_text(encoding="utf-8")
    assert 'choices=["binary", "hybrid-rank"]' in source
    assert "def marginal_gain_ranks" in source
    assert "smooth_l1_loss" in source
    assert '"loss_mode": args.loss_mode' in source


def test_tangent_student_canonicalizes_sparse_accumulation_order():
    source = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "hf_temporal_tangent_distilled.py"
    ).read_text(encoding="utf-8")
    assert "selected[row].to(hidden.device).sort().values" in source


def test_selective_fallback_is_reproducible_and_canonical():
    source = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "hf_selective_fallback.py"
    ).read_text(encoding="utf-8")
    assert 'parser.add_argument("--revision")' in source
    assert '"model_revision"' in source
    assert '"tokenizer_revision"' in source
    assert '"seed": args.seed' in source
    assert "selected[row].sort().values" in source


def test_sparse_evaluators_canonicalize_selected_neuron_order():
    experiments = Path(__file__).resolve().parents[1] / "experiments"
    evaluators = [
        "hf_binary_address.py",
        "hf_causal_state_codebook.py",
        "hf_decode_rollout_hot_residual.py",
        "hf_decode_state_trained.py",
        "hf_distilled_address.py",
        "hf_global_lowbit_decode.py",
        "hf_greedy_end_to_end.py",
        "hf_hot_residual_sparse.py",
        "hf_hybrid_address.py",
        "hf_logit_tangent_oracle.py",
        "hf_lowbit_residual_pursuit.py",
        "hf_oracle_physical_cache_frontier.py",
        "hf_predictive_correction.py",
        "hf_risk_controlled_fallback.py",
        "hf_selective_fallback.py",
        "hf_tangent_distilled_address.py",
        "hf_temporal_address_cache.py",
        "hf_temporal_logit_tangent_cache.py",
        "hf_temporal_predictive_coding.py",
        "hf_temporal_tangent_distilled.py",
    ]
    for name in evaluators:
        source = (experiments / name).read_text(encoding="utf-8")
        assert ".sort().values" in source, name
