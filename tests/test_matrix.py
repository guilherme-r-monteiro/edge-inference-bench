from pathlib import Path

from scripts.run_matrix import Run, command, plan, summarize


def test_command_lists_arguments_in_order() -> None:
    run = Run("x", "m.gguf", {"threads": 4, "prompt": 512})
    assert command(Path("bench"), run) == ["bench", "m.gguf", "--threads", "4", "--prompt", "512"]


def test_summary_takes_the_median() -> None:
    rows = [
        {"load_seconds": s, "rss_kb_after_load": 1, "rss_kb_peak": 1, "prefill_tokens_per_second": 1, "decode_tokens_per_second": d}
        for s, d in ((1.0, 10.0), (3.0, 30.0), (2.0, 20.0))
    ]
    summary = summarize("a", rows)
    assert summary["load_seconds"] == 2.0 and summary["decode_tokens_per_second"] == 20.0


def test_plan_has_unique_labels_and_one_long_session() -> None:
    labels = [run.label for run in plan()]
    assert len(labels) == len(set(labels)) and labels.count("long_session") == 1
