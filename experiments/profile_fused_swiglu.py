"""Single-region Nsight Compute target for dense and fused SwiGLU kernels."""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM

from depth_router.fused_swiglu import PrepackedSwiGLU, TritonPrepackedSwiGLU


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision", default="7ae557604adf67be50417f59c2c2f167def9a775")
    parser.add_argument("--method", choices=("dense", "triton"), required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--fraction", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda")
    model.eval()
    mlp = model.model.layers[-1].mlp
    width = mlp.gate_proj.out_features
    retained = width if args.method == "dense" else round(width * args.fraction)
    indices = torch.arange(retained, device="cuda")
    module_type = PrepackedSwiGLU if args.method == "dense" else TritonPrepackedSwiGLU
    module = module_type(mlp.gate_proj, mlp.up_proj, mlp.down_proj, indices)
    x = torch.randn(args.batch, mlp.gate_proj.in_features, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        for _ in range(5):
            module(x)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        module(x)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
