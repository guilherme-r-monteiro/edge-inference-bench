import json
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPEATS = 3
THREAD_COUNTS = (1, 2, 4, 8, 12)
MODELS = {"q4_k_m": "models/q4_k_m.gguf", "q8_0": "models/q8_0.gguf"}
LOAD_MODES = {"none": 0, "mmap": 1, "direct_io": 4}
DEFAULT_THREADS = 4
LONG_GENERATE = 3000
LONG_CONTEXT = 4096
REPORT_EVERY = 500


@dataclass(frozen=True)
class Run:
    label: str
    model: str
    arguments: dict[str, int] = field(default_factory=dict)


def command(binary: Path, run: Run) -> list[str]:
    parts = [str(binary), run.model]
    for key, value in run.arguments.items():
        parts += [f"--{key}", str(value)]
    return parts


def execute(binary: Path, run: Run) -> dict[str, object]:
    completed = subprocess.run(command(binary, run), capture_output=True, text=True, check=True)
    return json.loads(completed.stdout.strip().splitlines()[-1])


def median_of(rows: list[dict[str, object]], key: str) -> float:
    return statistics.median(float(row[key]) for row in rows)


def summarize(label: str, rows: list[dict[str, object]]) -> dict[str, object]:
    keys = ("load_seconds", "rss_kb_after_load", "rss_kb_peak", "prefill_tokens_per_second", "decode_tokens_per_second")
    summary = {key: median_of(rows, key) for key in keys}
    summary["label"] = label
    summary["window_rates"] = rows[len(rows) // 2].get("window_rates", [])
    return summary


def plan() -> list[Run]:
    runs = [Run(f"threads_{t}", MODELS["q4_k_m"], {"threads": t, "prompt": 512, "generate": 128}) for t in THREAD_COUNTS]
    runs += [Run(f"quant_{name}", path, {"threads": DEFAULT_THREADS, "prompt": 512, "generate": 128}) for name, path in MODELS.items()]
    runs += [
        Run(f"load_{name}", MODELS["q8_0"], {"threads": DEFAULT_THREADS, "prompt": 64, "generate": 8, "load-mode": mode})
        for name, mode in LOAD_MODES.items()
    ]
    runs.append(
        Run(
            "long_session",
            MODELS["q4_k_m"],
            {"threads": DEFAULT_THREADS, "prompt": 512, "generate": LONG_GENERATE, "context": LONG_CONTEXT, "report-every": REPORT_EVERY},
        )
    )
    return runs


def main(binary: Path, output: Path, extra: dict[str, int]) -> None:
    results = []
    for run in plan():
        merged = Run(run.label, run.model, {**run.arguments, **extra})
        rows = [execute(binary, merged) for _ in range(REPEATS)]
        results.append(summarize(run.label, rows))
        sys.stdout.write(f"{run.label} decode={results[-1]['decode_tokens_per_second']:.1f} prefill={results[-1]['prefill_tokens_per_second']:.1f}\n")
        sys.stdout.flush()
    output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    gpu_layers = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    main(Path(sys.argv[1]), Path(sys.argv[2]), {"gpu-layers": gpu_layers})
