# Real pretrained-model gate

The toy experiments have established two things worth taking to a real model:

1. a strict output-distortion oracle can often reproduce a dense cold MLP with a
   small subset of physical pages;
2. function-preserving neuron repacking can reduce the number of physical pages
   needed by co-demanding neurons.

The next gate is deliberately narrower than a GPU speed claim.

## Gate A: exact compiler transformation on a pretrained SwiGLU model

For a Llama/Qwen/Mistral-style MLP

```text
u = silu(W_gate h) * (W_up h)
y = W_down u
```

any permutation P of the intermediate neurons is an exact symmetry when applied
consistently:

```text
W_gate' = P W_gate
W_up'   = P W_up
W_down' = W_down P^-1
```

The dense function is unchanged.

DepthRouter uses held-out activation traces to learn P from neuron co-demand and
then rewrites the physical layout of the three matrices.

This is compiler work, not fine-tuning.

## Run

On the Spark or another machine with enough memory:

```bash
python -m pip install -e ".[hf]"

python experiments/hf_swiglu_compile.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --units-per-page 256 \
  --mass-target 0.95 \
  --device cuda \
  --dtype bfloat16 \
  --json-out real_model_gate.json
```

For a serious run, pass a conversation/session-level text corpus:

```bash
python experiments/hf_swiglu_compile.py \
  --model /models/my-local-model \
  --text-file calibration.jsonl \
  --layer 12 \
  --units-per-page 256 \
  --mass-target 0.95 \
  --device cuda \
  --dtype bfloat16 \
  --save-compiled compiled-model \
  --json-out real_model_gate.json
```

Each JSONL row may be either plain text or:

```json
{"text": "the conversation or prompt"}
```

## Pass conditions

The first real-model gate passes only when all of these are true:

- the pretrained model's logits are preserved to numerical precision after the
  physical rewrite;
- held-out traces show a repeatable reduction in pages required to cover the
  same co-demand mass;
- the gain survives multiple disjoint calibration/test conversation splits.

A pass here establishes a **layout result**, not an inference speedup.

## Gate B: behavioral page sparsity

After Gate A passes, replace the structural mass metric with the stronger target:

```text
P(KL(p_dense || p_paged) <= delta) >= 1 - alpha
```

using exact cold page contributions on a manageable evaluation slice.

The key question is whether the pretrained MLP still has a small
behaviorally-sufficient working set after compilation.

## Gate C: measured memory traffic

Only then build the page-selective kernel and measure:

- HBM / memory-controller bytes;
- L2 hit rate;
- decode latency;
- page-fault coalescing across tokens.

The headline metric remains:

```text
cross-entropy improvement in nats
---------------------------------
coalesced bytes transferred
```

If the compiler improves structural locality but actual traffic does not fall,
the systems thesis fails.
