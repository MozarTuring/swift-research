#!/usr/bin/env python3
"""
SWIFT Baseline Experiment Runner

Runs SWIFT baseline experiments with configurable parameters.
Wraps the SWIFT evaluation modules for systematic experimentation.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

# Add SWIFT directory to Python path for imports
SCRIPT_DIR = Path(__file__).parent.resolve()
SWIFT_DIR = SCRIPT_DIR / "SWIFT"
if str(SWIFT_DIR) not in sys.path:
    sys.path.insert(0, str(SWIFT_DIR))


def parse_args():
    parser = argparse.ArgumentParser(description="SWIFT Baseline Experiment Runner")
    
    # Model configuration
    parser.add_argument("--model", type=str, required=True,
                        help="Model name/path (e.g., meta-llama/Llama-3-8B)")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Local path to model (overrides --model if set)")
    
    # Experiment configuration
    parser.add_argument("--experiment-type", type=str, default="baseline",
                        choices=["baseline", "swift", "comparison"],
                        help="Type of experiment to run")
    parser.add_argument("--num-runs", type=int, default=3,
                        help="Number of repeated runs for averaging")
    
    # Output configuration
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save results")
    
    # GPU configuration
    parser.add_argument("--gpus", type=int, default=1,
                        help="Number of GPUs to use")
    
    # SWIFT hyperparameters
    parser.add_argument("--opt-interval", type=int, default=1,
                        help="Optimization interval")
    parser.add_argument("--bayes-interval", type=int, default=25,
                        help="Bayesian optimization interval")
    parser.add_argument("--max-opt-iter", type=int, default=1000,
                        help="Maximum optimization iterations")
    parser.add_argument("--max-tolerance-iter", type=int, default=300,
                        help="Maximum tolerance iterations")
    parser.add_argument("--max-score", type=float, default=0.93,
                        help="Maximum score threshold")
    parser.add_argument("--context-window", type=int, default=50,
                        help="Context window size")
    parser.add_argument("--skip-ratio", type=float, default=0.45,
                        help="Layer skip ratio")
    
    # Inference parameters
    parser.add_argument("--task-name", type=str, default="cnndm",
                        choices=["cnndm", "humaneval"],
                        help="Evaluation task")
    parser.add_argument("--data-num", type=int, default=100,
                        help="Number of data samples")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.85,
                        help="Top-p sampling threshold")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Maximum new tokens to generate")
    parser.add_argument("--seed", type=int, default=2024,
                        help="Random seed")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float32", "float64", "float16", "bfloat16"],
                        help="Data type for model")
    
    return parser.parse_args()


def get_model_id(model_path):
    """Extract model ID from path for naming."""
    if model_path:
        return Path(model_path).name
    return Path(model_path).name if model_path else "unknown"


def run_baseline_experiment(model_path, model_id, output_dir, args, run_idx):
    """Run baseline (vanilla) inference."""
    from evaluation_llama import inference_baseline
    
    run_output = Path(output_dir) / f"baseline_run{run_idx}"
    run_output.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Running Baseline (Run {run_idx})")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    inference_baseline.run_inference(
        model_path=model_path,
        model_id=model_id,
        task_name=args.task_name,
        data_num=args.data_num,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed + run_idx,
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
        output_dir=str(run_output)
    )
    
    elapsed = time.time() - start_time
    print(f"Baseline run {run_idx} completed in {elapsed:.2f}s")
    
    return {
        "type": "baseline",
        "run": run_idx,
        "elapsed_time": elapsed,
        "output_dir": str(run_output)
    }


def run_swift_experiment(model_path, model_id, output_dir, args, run_idx):
    """Run SWIFT-accelerated inference."""
    from evaluation_llama import inference_swift
    
    run_output = Path(output_dir) / f"swift_run{run_idx}"
    run_output.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Running SWIFT (Run {run_idx})")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    inference_swift.run_inference(
        model_path=model_path,
        model_id=model_id,
        task_name=args.task_name,
        data_num=args.data_num,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed + run_idx,
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
        context_window=args.context_window,
        opt_interval=args.opt_interval,
        bayes_interval=args.bayes_interval,
        max_opt_iter=args.max_opt_iter,
        max_tolerance_iter=args.max_tolerance_iter,
        max_score=args.max_score,
        skip_ratio=args.skip_ratio,
        optimization=True,
        bayes=True,
        output_dir=str(run_output)
    )
    
    elapsed = time.time() - start_time
    print(f"SWIFT run {run_idx} completed in {elapsed:.2f}s")
    
    return {
        "type": "swift",
        "run": run_idx,
        "elapsed_time": elapsed,
        "output_dir": str(run_output)
    }


def save_experiment_config(args, output_dir, model_id):
    """Save experiment configuration to JSON."""
    config = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "model_path": args.model_path,
        "model_id": model_id,
        "experiment_type": args.experiment_type,
        "num_runs": args.num_runs,
        "gpus": args.gpus,
        "task_name": args.task_name,
        "data_num": args.data_num,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "dtype": args.dtype,
        "swift_params": {
            "opt_interval": args.opt_interval,
            "bayes_interval": args.bayes_interval,
            "max_opt_iter": args.max_opt_iter,
            "max_tolerance_iter": args.max_tolerance_iter,
            "max_score": args.max_score,
            "context_window": args.context_window,
            "skip_ratio": args.skip_ratio
        }
    }
    
    config_path = Path(output_dir) / "experiment_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nExperiment config saved to: {config_path}")


def main():
    args = parse_args()
    
    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine model path and ID
    model_path = args.model_path or args.model
    model_id = get_model_id(model_path)
    
    # Save experiment configuration
    save_experiment_config(args, output_dir, model_id)
    
    print(f"\n{'#'*60}")
    print(f"# SWIFT Experiment Runner")
    print(f"# {'#'*56}")
    print(f"# Model: {args.model}")
    print(f"# Experiment Type: {args.experiment_type}")
    print(f"# Number of Runs: {args.num_runs}")
    print(f"# Output Directory: {output_dir}")
    print(f"# {'#'*60}\n")
    
    # Run experiments
    results = []
    
    if args.experiment_type in ["baseline", "comparison"]:
        for i in range(args.num_runs):
            result = run_baseline_experiment(model_path, model_id, output_dir, args, i)
            results.append(result)
    
    if args.experiment_type in ["swift", "comparison"]:
        for i in range(args.num_runs):
            result = run_swift_experiment(model_path, model_id, output_dir, args, i)
            results.append(result)
    
    # Save results summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_runs": len(results),
        "results": results
    }
    
    summary_path = Path(output_dir) / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Experiment completed!")
    print(f"Results saved to: {output_dir}")
    print(f"Summary: {summary_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
