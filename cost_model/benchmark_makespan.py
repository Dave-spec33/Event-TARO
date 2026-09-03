#!/usr/bin/env python3

import os

# Match the validated GRPO environment.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_HOME"] = "/usr/local/cuda-12.8"
os.environ["PATH"] = (
    "/usr/local/cuda-12.8/bin:"
    + os.environ.get("PATH", "")
)
os.environ["LIBRARY_PATH"] = (
    "/usr/local/cuda-12.8/targets/x86_64-linux/lib:"
    + os.environ.get("LIBRARY_PATH", "")
)
os.environ["LD_LIBRARY_PATH"] = (
    "/usr/local/cuda-12.8/targets/x86_64-linux/lib:"
    + os.environ.get("LD_LIBRARY_PATH", "")
)

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"


import argparse
import csv
import json
import statistics
import time
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt-file",
        default="/root/autodl-tmp/taro/analysis/taro_step1/selected_prompts.json",
    )

    parser.add_argument(
        "--model-path",
        default="/root/autodl-tmp/taro/models/Qwen3-1.7B",
    )

    parser.add_argument(
        "--output-dir",
        default="/root/autodl-tmp/taro/analysis/taro_step1",
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
    )

    return parser.parse_args()


def format_prompt(prompt, tokenizer):

    if isinstance(prompt, str):
        return prompt

    if isinstance(prompt, list):
        try:
            return tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )

    return str(prompt)


def make_sampling_params(length):
    """
    Force exact decode length.

    This removes natural EOS/stochastic length variation and lets
    the experiment isolate system scheduling / batching effects.
    """

    return SamplingParams(
        temperature=0.0,
        max_tokens=length,
        ignore_eos=True,
    )


def run_once(
    llm,
    prompts,
    lengths,
):
    params = [
        make_sampling_params(length)
        for length in lengths
    ]

    start = time.perf_counter()

    outputs = llm.generate(
        prompts,
        params,
        use_tqdm=False,
    )

    elapsed = time.perf_counter() - start

    actual_lengths = [
        len(out.outputs[0].token_ids)
        for out in outputs
    ]

    return elapsed, actual_lengths


def benchmark(
    llm,
    prompts,
    lengths,
    repeats,
):

    # One warmup execution for this shape.
    run_once(
        llm,
        prompts,
        lengths,
    )

    times = []
    actual_lengths = None

    for rep in range(repeats):

        elapsed, generated = run_once(
            llm,
            prompts,
            lengths,
        )

        times.append(elapsed)
        actual_lengths = generated

        print(
            f"repeat={rep + 1} "
            f"time={elapsed:.3f}s "
            f"lengths={generated}"
        )

    return {
        "median": statistics.median(times),
        "mean": statistics.mean(times),
        "times": times,
        "actual_lengths": actual_lengths,
    }


def main():

    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=False,
    )

    with open(
        args.prompt_file,
        "r",
        encoding="utf-8",
    ) as f:
        raw_prompts = json.load(f)

    prompts = [
        format_prompt(
            item["prompt"],
            tokenizer,
        )
        for item in raw_prompts
    ]

    # Need distinct prompts to avoid benchmarking identical requests.
    if len(prompts) < 10:
        raise RuntimeError(
            "Need at least 10 distinct prompts."
        )

    print(f"Loaded {len(prompts)} prompts.")

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=False,

        seed=42,

        gpu_memory_utilization=0.45,
        max_model_len=8704,
        max_num_batched_tokens=4096,
        max_num_seqs=16,

        enable_chunked_prefill=True,
        enforce_eager=False,

        generation_config="vllm",

        compilation_config={
            "level": 3,
            "cudagraph_capture_sizes": [
                1, 2, 4, 8, 16
            ],
        },
    )

    print("\nEngine warmup...")

    run_once(
        llm,
        [prompts[0]],
        [128],
    )

    # --------------------------------------------------------
    # Experiment design
    # --------------------------------------------------------

    # Candidate is always the SAME prompt.
    candidate_prompt = prompts[0]

    candidate_lengths = [
        1024,
        4096,
        8192,
    ]

    # Existing workload:
    # number of concurrent 8192-token generations.
    state_sizes = [
        0,
        1,
        2,
        4,
        8,
    ]

    # Use other prompts as background jobs.
    background_pool = prompts[1:10]

    # --------------------------------------------------------
    # Benchmark base states once.
    # --------------------------------------------------------

    base_results = {
        0: {
            "median": 0.0,
            "actual_lengths": [],
        }
    }

    for n in state_sizes:

        if n == 0:
            continue

        state_prompts = background_pool[:n]

        state_lengths = [
            8192
        ] * n

        print()
        print("=" * 70)
        print(
            f"BASE STATE: {n} x 8192"
        )
        print("=" * 70)

        base_results[n] = benchmark(
            llm,
            state_prompts,
            state_lengths,
            args.repeats,
        )

    rows = []

    # --------------------------------------------------------
    # Add the same candidate under different states.
    # --------------------------------------------------------

    for candidate_length in candidate_lengths:

        # First measure standalone cost explicitly.
        print()
        print("=" * 70)
        print(
            f"CANDIDATE STANDALONE: "
            f"{candidate_length}"
        )
        print("=" * 70)

        standalone = benchmark(
            llm,
            [candidate_prompt],
            [candidate_length],
            args.repeats,
        )

        standalone_time = standalone[
            "median"
        ]

        for n in state_sizes:

            base = base_results[n]

            if n == 0:
                combined = standalone

            else:

                state_prompts = (
                    background_pool[:n]
                )

                combined_prompts = (
                    state_prompts
                    + [candidate_prompt]
                )

                combined_lengths = (
                    [8192] * n
                    + [candidate_length]
                )

                print()
                print("=" * 70)
                print(
                    f"STATE={n}x8192 "
                    f"+ CANDIDATE={candidate_length}"
                )
                print("=" * 70)

                combined = benchmark(
                    llm,
                    combined_prompts,
                    combined_lengths,
                    args.repeats,
                )

            delta = (
                combined["median"]
                - base["median"]
            )

            marginal_ratio = (
                delta / standalone_time
                if standalone_time > 0
                else 0.0
            )

            rows.append(
                {
                    "background_long_jobs": n,

                    "candidate_tokens":
                        candidate_length,

                    "base_makespan_s":
                        base["median"],

                    "combined_makespan_s":
                        combined["median"],

                    "delta_makespan_s":
                        delta,

                    "candidate_standalone_s":
                        standalone_time,

                    "marginal_cost_ratio":
                        marginal_ratio,
                }
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    csv_path = (
        output_dir
        / "makespan_v2_results.csv"
    )

    with open(
        csv_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 80)
    print("TARO MARGINAL MAKESPAN V2")
    print("=" * 80)

    for row in rows:

        print(
            f"state={row['background_long_jobs']:2d} | "
            f"candidate={row['candidate_tokens']:4d} | "
            f"base={row['base_makespan_s']:7.2f}s | "
            f"combined={row['combined_makespan_s']:7.2f}s | "
            f"delta={row['delta_makespan_s']:7.2f}s | "
            f"standalone={row['candidate_standalone_s']:7.2f}s | "
            f"ratio={row['marginal_cost_ratio']:.3f}"
        )

    print()
    print("Saved to:", csv_path)


if __name__ == "__main__":
    main()