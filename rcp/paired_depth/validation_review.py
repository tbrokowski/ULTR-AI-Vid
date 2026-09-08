"""Restricted validation-only paired analysis; exports aggregate results only."""
import argparse
import json
from pathlib import Path

from ultrai.paired_depth.manifest import write_json
from ultrai.paired_depth.metrics import paired_bootstrap


def indexed(run, view):
    rows = [r for r in json.loads((run / "valid_predictions.json").read_text()) if r["view"] == view]
    result = {r["patient"]: r for r in rows}
    if len(rows) != len(result) or not result:
        raise ValueError("Missing or duplicated validation patients")
    return result


def review(root, output, samples=2000):
    root, output = Path(root), Path(output)
    arms = ("source15", "both_supervised", "consistency", "dann", "conditional", "full")
    results = {"scope": "partition 0 seed 42 validation only", "test_data_read": False,
               "interval_limitation": "Conditional on validation-selected checkpoints; does not include selection or training variability",
               "bootstrap_samples": samples, "bootstrap_seed": 20260906,
               "runs": {}, "paired_comparisons": {}, "complete": False}
    for arm in arms:
        run = root / f"{arm}-p0-s42"
        progress = json.loads((run / "progress.json").read_text())
        if progress["status"] != "complete" or len(progress["history"]) != 6 or progress["updates"] != 12:
            raise ValueError("Comparison did not complete the frozen schedule")
        results["runs"][arm] = json.loads((run / "valid_metrics.json").read_text())
    contrasts = [("source15", arm, view) for arm in ("both_supervised", "consistency", "full")
                 for view in ("matched5", "all15")]
    contrasts.append(("both_supervised", "dann", "matched5"))
    for baseline, candidate, view in contrasts:
        a, b = [indexed(root / f"{arm}-p0-s42", view) for arm in (baseline, candidate)]
        if a.keys() != b.keys() or any(a[p]["label"] != b[p]["label"] for p in a):
            raise ValueError("Paired comparison has differing cohorts or labels")
        patients = sorted(a)
        interval = paired_bootstrap([a[p]["label"] for p in patients],
                                    [a[p]["probability"] for p in patients],
                                    [b[p]["probability"] for p in patients], samples=samples)
        name = f"{candidate}-minus-{baseline}:{view}"
        results["paired_comparisons"][name] = interval
        write_json(output, results)
        print(json.dumps({"comparison": name, "difference": interval["difference"]}), flush=True)
    results["complete"] = True
    write_json(output, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=2000)
    args = parser.parse_args()
    review(args.run_root, args.output, args.samples)
