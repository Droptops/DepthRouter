# Enterprise working-set frontier: execution and interpretation

This is an **exploratory synthetic workload** with 100 prompts each for classification, extraction and tool selection. Prompts are deterministic templates, not representative production samples. No claim about enterprise generalization is warranted.

## Reproduce
Install `pip install -e ".[hf]"` on a machine with sufficient CPU RAM/GPU capacity.

```bash
python experiments/enterprise_tasks.py --examples-per-task 100 --out enterprise_300.jsonl
python experiments/hf_enterprise_working_set_trace.py \
  --model Qwen/Qwen2.5-0.5B-Instruct --layer 23 --steps 3 \
  --examples-per-task 100 --page-units 128 --active-fraction 0.10 \
  --train-out enterprise_train.jsonl --test-out enterprise_test.jsonl \
  --metadata-out enterprise_trace_meta.json
python experiments/enterprise_working_set_replay.py \
  --train enterprise_train.jsonl --test enterprise_test.jsonl \
  --capacity 16 --page-bytes 131072 --transition-threshold 0.1 \
  --out enterprise_replay.json
```

**Important:** `--page-bytes 131072` is a placeholder, NOT a measured physical page size. It must be replaced with a documented storage layout before byte comparisons have engineering meaning.

The dense trace uses top-k activation-magnitude × down-projection norm as a *demand proxy*. This does not prove these pages are sufficient for accurate sparse logits. A page is 128 adjacent neurons in the unpermuted weight matrix; no physical paging kernel is used. Dense forward captures demand; the transition signal is a one-step-lagged hidden delta.

## What this first run can and cannot show

Can: train/test-disjoint logical page-fault counts, frequency overlap, cache-policy replay behavior, and whether a lagged transition proxy is promising.

Cannot: the minimum quality-preserving parameter count, task accuracy after pruning, KL, top-1 agreement, measured physical HBM traffic, physical page transfers, or latency savings. These fields must remain null until separately measured.

The current transition policy is a deliberately simple *eviction baseline*, not a trained controller. In particular, if each step requests as many pages as fit in the cache, eviction choices have little freedom; if this makes transition and previous-step policies tie, treat it as an informative failure rather than claim a gain.

## Next gates before promotion
1. Run the existing tests plus the 300-example Qwen collector on a real-model runner; publish raw JSONL and provenance.
2. Evaluate per-task and pooled overlap, transfer estimates, and out-of-domain held-out prompts; tune thresholds on calibration only.
3. Evaluate actual sparse/fallback logits and free-running task success, including a sweep over working-set budgets. Record the smallest **observed passing** configuration, not a mathematical minimum.
4. Implement a physical offload/selective kernel and profile device-to-device or host-to-device transfer bytes and latency; report physical measurements separately from logical replay.
5. Promote only if mean KL < 0.01, top-1 agreement >= 99%, task accuracy passes preregistered gates, and physical bytes/token improve over previous-token residency including detector overhead.

No automatic merge while those gates are unfulfilled.
