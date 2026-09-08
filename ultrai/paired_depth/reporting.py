"""Export aggregate results only. Patient predictions stay in restricted storage."""
import argparse
import json
from pathlib import Path

import numpy as np

from .manifest import digest, write_json
from .metrics import paired_bootstrap, scores


def freeze(config_paths, output, selection):
    from .engine import load_config
    configs = [load_config(p) for p in config_paths]
    if Path(output).exists():
        raise FileExistsError("A frozen test protocol cannot be overwritten")
    for cfg in configs:
        root = Path(cfg["output"])
        if not (root / "valid_metrics.json").exists() or not (root / "best.pt").exists():
            raise ValueError("Every test run must have validation metrics and a best checkpoint")
        if (root / "test_predictions.json").exists():
            raise ValueError("Test data was already evaluated before protocol freeze")
    record = {"schema": 1, "selection": selection,
              "checkpoint_sha256": [digest(Path(c["output"]) / "best.pt") for c in configs],
              "resolved_configs": configs,
              "validation_metrics": [json.loads((Path(c["output"]) / "valid_metrics.json").read_text()) for c in configs]}
    write_json(output, record)
    return record


def aggregate(run_pairs, samples=2000):
    result = {}
    for view in ("matched5", "matched15", "all15"):
        pairs, ids, reference_labels, run_ids = [], None, None, []
        for baseline_dir, candidate_dir in run_pairs:
            maps = []
            for directory in (baseline_dir, candidate_dir):
                records = json.loads((Path(directory) / "test_predictions.json").read_text())
                filtered = [r for r in records if r["view"] == view]
                mapping = {r["patient"]: r for r in filtered}
                if len(mapping) != len(filtered):
                    raise ValueError("Duplicate patient prediction within an evaluation view")
                maps.append(mapping)
            current_ids = sorted(maps[0])
            if current_ids != sorted(maps[1]) or (ids is not None and current_ids != ids):
                raise ValueError("All model runs must use the same patient cohort; do not silently intersect")
            ids = current_ids
            labels = [maps[0][p]["label"] for p in ids]
            if labels != [maps[1][p]["label"] for p in ids] or (reference_labels is not None and labels != reference_labels):
                raise ValueError("Inconsistent patient labels across runs")
            reference_labels = labels
            identifiers = [(m[ids[0]]["partition"], m[ids[0]]["seed"]) for m in maps]
            if identifiers[0] != identifiers[1]:
                raise ValueError("Baseline and candidate partition/seed must match")
            run_ids.append({"partition": identifiers[0][0], "seed": identifiers[0][1]})
            pairs.append([[m[p]["probability"] for p in ids] for m in maps])
        if not pairs:
            raise ValueError("No paired completed model runs")
        if len({(r["partition"], r["seed"]) for r in run_ids}) != len(run_ids):
            raise ValueError("Duplicate partition/seed model comparison")
        arrays = np.asarray(pairs)
        estimate = paired_bootstrap(reference_labels, arrays[:, 0], arrays[:, 1], samples=samples)
        per_run = [{**identity, "baseline": scores(reference_labels, a), "candidate": scores(reference_labels, b)}
                   for identity, (a, b) in zip(run_ids, arrays)]
        variation = {}
        for fixed, varied in (("seed", "partition"), ("partition", "seed")):
            variation[f"across_{varied}_within_{fixed}"] = {
                str(value): {arm: {metric: float(np.std([r[arm][metric] for r in per_run if r[fixed] == value], ddof=1))
                                   if sum(r[fixed] == value for r in per_run) > 1 else None
                                   for metric in ("auroc", "average_precision")} for arm in ("baseline", "candidate")}
                for value in sorted({r[fixed] for r in per_run})}
        result[view] = {"patients": len(ids), "runs": len(pairs), "bootstrap": estimate, "per_run": per_run,
                        "training_variation": variation}
    ci = result["matched5"]["bootstrap"]["difference"]["auroc"]["ci95"]
    result["interpretation"] = {"cross_depth_inconclusive": ci[0] <= 0 <= ci[1],
                                "stable15_point_estimate": result["all15"]["bootstrap"]["difference"]["auroc"]["estimate"] >= -0.01,
                                "absolute15_reaches_082": result["all15"]["bootstrap"]["candidate"]["auroc"]["estimate"] >= 0.82,
                                "south_africa_evaluated": False}
    return result


def markdown(summary, aggregate_results=None):
    """Render only supplied audit information and computed metrics."""
    lines = ["# Paired-depth study output", "",
             "## Input audit", "",
             f"Dataset: {summary.get('dataset', 'unspecified')}.",
             f"Revision: {summary.get('revision', 'unspecified')}.",
             f"Revision verified: {summary.get('revision_verified', False)}.",
             f"Eligible acquisition pairs: {summary.get('eligible_pairs', 'unspecified')}.",
             f"Patients with pairs: {summary.get('paired_patients', 'unspecified')}.", ""]
    if aggregate_results is None:
        lines += ["No model evaluation was supplied to this report.", ""]
    else:
        lines += ["## Paired test comparisons", "",
                  "| View | Metric | Baseline | Candidate | Difference [95% CI] |",
                  "|---|---|---:|---:|---:|"]
        for view in ("matched5", "matched15", "all15"):
            values = aggregate_results[view]["bootstrap"]
            for metric in ("auroc", "average_precision"):
                difference = values["difference"][metric]
                lines.append(
                    f"| {view} | {metric} | {values['baseline'][metric]['estimate']:.4f} | "
                    f"{values['candidate'][metric]['estimate']:.4f} | "
                    f"{difference['estimate']:+.4f} "
                    f"[{difference['ci95'][0]:+.4f}, {difference['ci95'][1]:+.4f}] |")
        lines += ["", "Metrics are means of per-run patient metrics, not an ensemble.",
                  "Bootstrap samples resample patients jointly across the included runs.",
                  "Per-run metrics and separate partition/seed variation are in aggregate_results.json.",
                  "Interpret these intervals conditional on the supplied trained models.", ""]
    lines += ["## Protocol", "",
              "The five training/validation partitions share one test cohort.",
              "Compare matched-site patient bags at each depth and all-site bags at 15 cm.",
              "Choose configurations using validation only and freeze checkpoints before test evaluation.",
              "Patient predictions and checkpoints remain in the study's restricted output directory.",
              "See docs/paired_depth/REPRODUCE.md for the complete workflow.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--summary", required=True)
    p.add_argument("--comparisons", help="JSON list of [baseline_run_directory, candidate_run_directory]")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    summary = json.loads(Path(a.summary).read_text())
    results = aggregate(json.loads(Path(a.comparisons).read_text())) if a.comparisons else None
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.md").write_text(markdown(summary, results))
    if results:
        write_json(output / "aggregate_results.json", results)
