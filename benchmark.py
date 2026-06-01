"""Benchmark TriageLLM critic latency: GPU-pinned vs CPU-pinned.

Runs N critique calls per placement and reports median / p95 / max.
Useful both for tuning critic_timeout_s in config.yaml and as a regression
test when changing model sizes.

Usage:
    .\\.venv\\Scripts\\python.exe benchmark.py
    .\\.venv\\Scripts\\python.exe benchmark.py --runs 20
    .\\.venv\\Scripts\\python.exe benchmark.py --json
"""
import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent))
from router_hook import critic_score  # noqa: E402

# Representative task/answer pair — moderate size, realistic critic workload.
TASK = (
    "Write a Python function that reads a CSV file and returns a list of "
    "dictionaries. Each row's first row should be used as headers. Handle "
    "missing files gracefully."
)
ANSWER = (
    "```python\n"
    "import csv\n"
    "from pathlib import Path\n\n"
    "def read_csv(path: str) -> list[dict]:\n"
    "    p = Path(path)\n"
    "    if not p.exists():\n"
    "        return []\n"
    "    with p.open() as f:\n"
    "        return list(csv.DictReader(f))\n"
    "```"
)


async def _time_call(cpu_only: bool) -> float:
    t0 = time.time()
    await critic_score(TASK, ANSWER, cpu_only=cpu_only, timeout_s=60)
    return time.time() - t0


async def bench(runs: int = 10) -> dict:
    results: dict = {"runs_per_placement": runs}

    for placement, cpu_only in [("CPU", True), ("GPU", False)]:
        # Warmup call first — don't count it
        try:
            await _time_call(cpu_only)
        except Exception as e:
            print(f"[{placement}] warmup failed: {e}", file=sys.stderr)

        latencies: list[float] = []
        for i in range(runs):
            try:
                latencies.append(await _time_call(cpu_only))
            except Exception as e:
                print(f"[{placement}] run {i}: {e}", file=sys.stderr)
                latencies.append(float("inf"))

        finite = [t for t in latencies if t != float("inf")]
        if not finite:
            results[placement] = {"error": "all runs failed"}
            continue
        results[placement] = {
            "median": round(statistics.median(finite), 2),
            "mean":   round(statistics.mean(finite), 2),
            "p95":    round(sorted(finite)[int(0.95 * len(finite)) - 1], 2) if len(finite) >= 4 else None,
            "min":    round(min(finite), 2),
            "max":    round(max(finite), 2),
            "n":      len(finite),
            "failures": runs - len(finite),
        }
    return results


def render_text(data: dict) -> str:
    lines = ["Critic latency benchmark", "=" * 60, ""]
    runs = data["runs_per_placement"]
    for placement in ("CPU", "GPU"):
        r = data.get(placement, {})
        if "error" in r:
            lines.append(f"{placement}: {r['error']}")
            continue
        lines.append(f"{placement} (n={r['n']}, failures={r['failures']})")
        lines.append(f"  median: {r['median']}s")
        lines.append(f"  mean:   {r['mean']}s")
        lines.append(f"  p95:    {r['p95']}s")
        lines.append(f"  min:    {r['min']}s")
        lines.append(f"  max:    {r['max']}s")
        lines.append("")
    lines.append(f"(Each placement ran {runs} times after a warmup call.)")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark TriageLLM")
    # Task-benchmark mode (Issue #24)
    p.add_argument("--tasks", action="store_true",
                   help="Run the per-machine task-type benchmark")
    p.add_argument("--quick", action="store_true",
                   help="(with --tasks) quick mode: 2 prompts/category")
    p.add_argument("--models", type=str, default=None,
                   help="(with --tasks) comma-separated model list "
                        "(default: all installed minus skip-list)")
    p.add_argument("--output", type=str, default="benchmark_results.json",
                   help="(with --tasks) priors JSON output path")
    # Critic-latency mode (original)
    p.add_argument("--runs", type=int, default=10,
                   help="(critic-latency mode) runs per placement (default: 10)")
    p.add_argument("--json", action="store_true", help="Emit JSON")
    return p


async def main_async() -> int:
    args = build_parser().parse_args()

    if args.tasks:
        import task_benchmark
        import benchmark_prompts
        mode = "quick" if args.quick else "full"
        priors = await task_benchmark.run_benchmark(
            benchmark_prompts.PROMPTS, mode=mode, override=args.models,
            output=args.output)
        print(task_benchmark.render_report(priors, json_mode=args.json))
        return 0

    data = await bench(args.runs)
    print(json.dumps(data, indent=2) if args.json else render_text(data))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main_async()))
