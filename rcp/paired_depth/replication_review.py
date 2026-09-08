"""Descriptive validation summaries; never pool partitions or read test data."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def describe(values):
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values),
            "minimum": min(values), "maximum": max(values)}


def summarize(run_root, seed=42, method="consistency"):
    run_root = Path(run_root)
    arms = ("source15", "both_supervised", method)
    result = {"seed": seed, "selected_method": method, "test_data_read": False,
              "interpretation": "Descriptive variation across five training/validation partitions. "
              "Validation cohorts differ; their patients are not pooled. Sample SD is not a "
              "patient-bootstrap confidence interval or variation across seeds. Metrics are "
              "conditional on validation-selected checkpoints.", "runs": {}, "arms": {}, "differences": {}}
    for partition in range(5):
        for arm in arms:
            name = f"{arm}-p{partition}-s{seed}"
            directory = run_root / name
            progress = json.loads((directory / "progress.json").read_text())
            if (progress["status"] != "complete" or len(progress["history"]) != 6
                    or progress["updates"] != 12):
                raise ValueError(f"Incomplete frozen comparison: {name}")
            metrics_path = directory / "valid_metrics.json"
            result["runs"][name] = {"validation": json.loads(metrics_path.read_text()),
                "metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
                "epochs": len(progress["history"]), "updates": progress["updates"],
                "amp_skipped_updates": progress["amp_skipped_updates"]}
    def values(arm, view, metric):
        return [result["runs"][f"{arm}-p{p}-s{seed}"]["validation"][view][metric] for p in range(5)]
    for arm in arms:
        result["arms"][arm] = {view: {metric: describe(values(arm, view, metric))
            for metric in ("auroc", "average_precision")}
            for view in ("matched5", "matched15", "all15")}
        result["arms"][arm]["all15_auroc_at_least_0_82_partitions"] = sum(
            x >= 0.82 for x in values(arm, "all15", "auroc"))
    for arm, baseline in (("both_supervised", "source15"), (method, "source15"), (method, "both_supervised")):
        comparison = {}
        for view in ("matched5", "matched15", "all15"):
            comparison[view] = {}
            for metric in ("auroc", "average_precision"):
                delta = [a-b for a, b in zip(values(arm, view, metric), values(baseline, view, metric))]
                comparison[view][metric] = {"by_partition": delta, **describe(delta)}
        comparison["stable15_partitions"] = sum(d >= -0.01 for d in comparison["all15"]["auroc"]["by_partition"])
        result["differences"][f"{arm}-minus-{baseline}"] = comparison
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", default="consistency", choices=("consistency", "dann", "conditional", "full"))
    args = parser.parse_args()
    result = summarize(args.run_root, args.seed, args.method)
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path = Path(args.output)
    if path.exists() and path.read_text() != rendered:
        raise FileExistsError("Preserve the existing summary; use a separate output for changed inputs")
    path.write_text(rendered)
    print(json.dumps({"arms": result["arms"], "differences": result["differences"]}, indent=2))
