import torch

from depth_router.predictive_working_set import (
    LowRankWorkingSetPredictor,
    aggregate_page_scores,
    contiguous_operator_pages,
    expand_page_selections,
    make_evaluation_manifest,
    predictor_footprint,
    row_set_metrics,
    topk_rows,
    validate_manifest,
)


def test_manifest_is_large_balanced_and_split_isolated() -> None:
    rows = make_evaluation_manifest()
    validate_manifest(rows)
    assert len(rows) == 896
    assert sum(row["split"] == "heldout" for row in rows) == 512


def test_topk_and_set_metrics_are_row_local() -> None:
    scores = torch.tensor([[0.0, 4.0, 3.0, 1.0], [5.0, 1.0, 2.0, 4.0]])
    predicted = topk_rows(scores, 0.5)
    oracle = torch.tensor([[1, 2], [0, 2]])
    metrics = row_set_metrics(predicted, oracle)
    torch.testing.assert_close(metrics["recall"], torch.tensor([1.0, 0.5]))
    torch.testing.assert_close(metrics["jaccard"], torch.tensor([1.0, 1.0 / 3.0]))


def test_operator_pages_expand_to_exact_75_percent() -> None:
    pages = contiguous_operator_pages(4864, 16)
    assert {page.numel() for page in pages} == {304}
    neuron_scores = torch.arange(2 * 4864, dtype=torch.float32).reshape(2, 4864)
    page_scores = aggregate_page_scores(neuron_scores, pages)
    selected_pages = topk_rows(page_scores, 0.75)
    expanded = expand_page_selections(selected_pages, pages)
    assert expanded.shape == (2, 3648)
    assert torch.all(expanded[:, 1:] > expanded[:, :-1])


def test_predictor_footprint_is_analytic_and_lightweight() -> None:
    predictor = LowRankWorkingSetPredictor(1792, 64, 4864)
    features = torch.randn(12, 1792)
    predictor.set_normalization(features)
    assert predictor(features).shape == (12, 4864)
    footprint = predictor_footprint(
        predictor,
        hidden_width=896,
        intermediate_width=4864,
    )
    assert footprint.parameters == predictor.parameter_count
    assert footprint.bytes_bfloat16 == 2 * footprint.parameters
    assert footprint.fraction_of_swiglu_macs < 0.04
