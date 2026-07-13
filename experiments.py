"""
Experiments
-----------
Entry point for E-GRM experiments.

    # dry-run (pipeline validation, no GPU)
    python experiments.py --benchmark rm-bench --max-samples 200 --dry-run

    # real GPU inference
    python experiments.py --benchmark rm-bench --model Qwen/Qwen2.5-7B --max-samples 100

    # 8-bit quantized (save VRAM)
    python experiments.py --benchmark rm-bench --model Qwen/Qwen2.5-7B --max-samples 100 --load-in-8bit
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from typing import List, Dict, Optional, Callable

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from egrm_uncertainty import ParallelDecoder, UncertaintyEstimator
from egrm_scorer import DiscriminativeScorer, ReasoningPath, HybridLoss
from egrm_router import CoTRouter, SFTExample
from egrm_pipeline import EGRMPipeline, EGRMReport


# ---- Benchmark registry ----

BENCHMARK_REGISTRY = {
    "rm-bench": {
        "path": "./data/rm_bench.jsonl",
        "category": "reward_modeling",
        "n_samples": 1000,
        "avg_difficulty": 0.45,
    },
    "rmb": {
        "path": "./data/rmb.jsonl",
        "category": "reward_modeling",
        "n_samples": 500,
        "avg_difficulty": 0.38,
    },
    "reward-bench": {
        "path": "./data/reward_bench.jsonl",
        "category": "reward_modeling",
        "n_samples": 800,
        "avg_difficulty": 0.52,
    },
    "math500": {
        "path": "./data/math500.jsonl",
        "category": "math_reasoning",
        "n_samples": 500,
        "avg_difficulty": 0.72,
    },
    "gsm8k": {
        "path": "./data/gsm8k.jsonl",
        "category": "math_reasoning",
        "n_samples": 1319,
        "avg_difficulty": 0.55,
    },
}


# ---- Real HuggingFace model loading ----

class RealModelSession:
    """Holds tokenizer + model for efficient reuse across inference calls."""

    def __init__(self, tokenizer, model, device: str):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self._call_count = 0

    def _extract_answer(self, text: str) -> str:
        """Extract final answer from model output."""
        # Try "Answer:" pattern first
        for delimiter in ["\nAnswer:", "\nanswer:", "\nThe answer is", "\nTherefore,"]:
            if delimiter in text:
                return text.split(delimiter)[-1].strip()
        # Fallback: last non-empty line
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        return lines[-1] if lines else text.strip()

    def _generate(self, prompt: str, temperature: float, top_p: float,
                  max_tokens: int) -> str:
        self._call_count += 1
        inputs = self.tokenizer(prompt, return_tensors="pt",
                                truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=(temperature > 0),
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id
                or self.tokenizer.pad_token_id or 0,
            )
        full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Strip prompt from output
        if full.startswith(prompt):
            full = full[len(prompt):]
        return full.strip()

    def direct_fn(self) -> Callable:
        """Return a model_fn compatible with EGRMPipeline.run()."""
        session = self

        def fn(prompt: str, temperature: float = 0.0, top_p: float = 1.0,
               max_tokens: int = 128) -> str:
            out = session._generate(prompt, temperature, top_p, max_tokens)
            return session._extract_answer(out)

        return fn

    def cot_fn(self) -> Callable:
        """Return a CoT model_fn (appends 'Let's think step by step')."""
        session = self

        def fn(prompt: str, temperature: float = 0.7, top_p: float = 0.95,
               max_tokens: int = 256) -> str:
            cot_prompt = prompt + "\nLet's think step by step."
            out = session._generate(cot_prompt, temperature, top_p, max_tokens)
            return out  # keep full CoT chain, router extracts answer

        return fn


def _load_hf_model(
    model_id: str,
    device: str = "cuda",
    dtype: str = "auto",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
) -> RealModelSession:
    """Load a HuggingFace LLM and return a session with tokenizer + model."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"  Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading model: {model_id}")
    kwargs: Dict = {"trust_remote_code": True}

    if load_in_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["bnb_4bit_compute_dtype"] = torch.float16
        print(f"  (4-bit quantized)")
    elif load_in_8bit:
        kwargs["load_in_8bit"] = True
        print(f"  (8-bit quantized)")
    elif dtype == "float16" or dtype == "half":
        kwargs["torch_dtype"] = torch.float16
        print(f"  (fp16)")
    elif dtype == "bfloat16":
        kwargs["torch_dtype"] = torch.bfloat16
        print(f"  (bf16)")

    if device == "cuda" and not load_in_8bit and not load_in_4bit:
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()

    # Warmup: one short generation to init CUDA
    try:
        test_input = tokenizer("Hello", return_tensors="pt")
        device_str = next(model.parameters()).device
        test_input = {k: v.to(device_str) for k, v in test_input.items()}
        with torch.no_grad():
            _ = model.generate(**test_input, max_new_tokens=2)
        print(f"  Model loaded on: {device_str}")
    except Exception:
        pass

    return RealModelSession(tokenizer, model, device)


# ---- Dry-run mock helpers ----

def make_mock_direct_model_fn(direct_ratio: float = 0.58, seed: int = 42):
    """Mock model: returns consistent answers with probability = direct_ratio."""
    rng = random.Random(seed)
    def fn(prompt: str, temperature: float = 0.0, top_p: float = 1.0,
           max_tokens: int = 128) -> str:
        is_consistent = rng.random() < direct_ratio
        noise = "" if is_consistent else f" variant_{rng.randint(1, 9)}"
        return f"Answer_{hash(prompt) % 1000}{noise}"
    return fn


def make_mock_cot_model_fn(seed: int = 42):
    """Mock CoT model: generates reasoning chain + final answer."""
    rng = random.Random(seed)
    def fn(prompt: str, temperature: float = 0.7, top_p: float = 0.95,
           max_tokens: int = 256) -> str:
        steps = rng.randint(2, 5)
        cot = []
        for i in range(steps):
            val = rng.randint(10, 99)
            op = rng.choice(["+", "*", "→"])
            cot.append(f"Step {i + 1}: Computed {val} {op} {rng.randint(1, 9)}")
        answer_val = rng.randint(100, 999)
        return "\n".join(cot) + f"\nAnswer: {answer_val}"
    return fn


def generate_synthetic_prompts(n: int, seed: int = 42) -> List[str]:
    rng = random.Random(seed)
    templates = [
        "Solve: If x + {a} = {b}, what is x?",
        "Calculate {a} * {b} + {c}",
        "Is {n} a prime number? Explain.",
        "Find the derivative of f(x) = x^{p} + {q}x",
        "If a train travels {d} km in {h} hours, what is its speed?",
        "What is the probability of rolling {n} on a fair die?",
    ]
    prompts = []
    for i in range(n):
        t = rng.choice(templates)
        prompt = t.format(
            a=rng.randint(10, 99), b=rng.randint(100, 999),
            c=rng.randint(1, 50), n=rng.randint(10, 99),
            p=rng.randint(2, 5), q=rng.randint(1, 9),
            d=rng.randint(100, 999), h=rng.randint(1, 5),
        )
        prompts.append(prompt)
    return prompts


def generate_synthetic_sft_samples(n: int, seed: int = 42) -> List[SFTExample]:
    rng = random.Random(seed)
    prompts = generate_synthetic_prompts(n, seed)
    samples = []
    for i, prompt in enumerate(prompts):
        ans = rng.randint(10, 999)
        samples.append(SFTExample(
            prompt=prompt,
            direct_answer=f"{ans}",
            cot_answer=f"Step 1: Analyze input\nStep 2: Compute result = {ans}\nAnswer: {ans}",
            final_answer=f"{ans}",
            category="math",
        ))
    return samples


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(description="E-GRM Experiments")
    parser.add_argument("--benchmark", type=str, default="rm-bench",
                        choices=list(BENCHMARK_REGISTRY.keys()))
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--parallel", type=int, default=5)
    parser.add_argument("--cot-paths", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="./results")
    parser.add_argument("--train", action="store_true",
                        help="Run full training pipeline (SFT + GRPO)")
    parser.add_argument("--verbose", action="store_true", default=True)

    # GPU / model args
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda, cpu, mps")
    parser.add_argument("--dtype", type=str, default="auto",
                        choices=["auto", "float16", "bfloat16", "half"],
                        help="Model dtype")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Load model in 8-bit (bitsandbytes required)")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load model in 4-bit (bitsandbytes required)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    bench = BENCHMARK_REGISTRY[args.benchmark]
    print("=" * 60)
    print("E-GRM Experiment Runner")
    print(f"  Benchmark: {args.benchmark}")
    print(f"  Model:     {args.model}")
    print(f"  Samples:   {args.max_samples}")
    print(f"  Threshold: {args.threshold}")
    print("=" * 60)

    # Load / generate data
    print("\n[1] Loading prompts ...")
    prompts = generate_synthetic_prompts(args.max_samples, args.seed)
    print(f"  Generated {len(prompts)} synthetic prompts")

    # Model setup
    print("\n[2] Setting up model ...")
    direct_model_fn = None
    cot_model_fn = None

    if args.dry_run:
        direct_model_fn = make_mock_direct_model_fn(direct_ratio=0.58, seed=args.seed)
        cot_model_fn = make_mock_cot_model_fn(seed=args.seed)
        print("  Using dry-run mock models")
    else:
        # Real HuggingFace model loading
        device = "cuda" if hasattr(args, "device") and args.device else "cuda"
        session = _load_hf_model(
            args.model,
            device=device,
            dtype=args.dtype,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
        )
        direct_model_fn = session.direct_fn()
        cot_model_fn = session.cot_fn()
        print(f"  Model ready. Direct fn + CoT fn extracted.")

    # Pipeline
    print("\n[3] Running E-GRM pipeline ...")
    pipeline = EGRMPipeline(
        consensus_threshold=args.threshold,
        num_parallel_runs=args.parallel,
        num_cot_paths=args.cot_paths,
    )

    report = pipeline.run(
        prompts, direct_model_fn, cot_model_fn, max_tokens=128, verbose=True,
    )

    out_path = os.path.join(args.output, f"egrm_{args.benchmark}_{int(time.time())}.json")
    pipeline.to_json(report, out_path)
    print(f"\n[Results saved: {out_path}]")
    pipeline.print_summary(report)

    # Optional training
    if args.train:
        print("\n[4] Training E-GRM ...")
        sft_samples = generate_synthetic_sft_samples(min(args.max_samples, 100), args.seed)

        def gen_paths_fn(p):
            paths = []
            for i in range(3):
                paths.append(ReasoningPath(
                    path_id=f"p{i}",
                    cot_text=f"Step reasoning for: {p[:50]}",
                    final_answer=f"Answer_{hash(p) % 100}",
                    tokens_generated=120 + i * 20,
                ))
            return paths

        train_result = pipeline.train(sft_samples, prompts[:50], gen_paths_fn)
        print(f"  SFT samples: {train_result['sft']['n_samples']}")
        print(f"  GRPO pairs:  {train_result['grpo']['grpo_pairs']}")


if __name__ == "__main__":
    main()
