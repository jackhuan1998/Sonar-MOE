
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import cmp_to_key
from pathlib import Path
from typing import Iterable

import torch

from ultralytics import YOLO


SUBSETS = {
    "S": "S",
    "E": "E",
    "C": "C",
    "S+E": "S+E",
    "S+C": "S+C",
    "E+C": "E+C",
}
DEFAULT_SAVINGS = [20.0, 40.0, 60.0]


@dataclass
class RunRecord:
    subset: str
    active: str
    epochs: int
    hours: float
    estimated_full_hours: float
    gpu_gb: float
    save_dir: str
    best: str
    last: str
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    map50: float | None = None
    map5095: float | None = None
    rank_score: float | None = None


@dataclass
class BudgetPlan:
    saving_percent: float
    initial_epochs: int
    keep: int
    epoch_cost: int
    full_epoch_cost: int
    actual_saving_percent: float

    @property
    def tag(self) -> str:
        return f"k{self.keep}_e{self.initial_epochs}"


@dataclass
class TrainJob:
    subset: str
    active: str
    epochs: int
    name: str
    model_source: str | Path
    resume: bool = False
    stop_epoch: int | None = None


def write_moe_config(active: str, config_path: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "# Auto-written by budget_expert_search.py\n"
        "# active supports aliases: S, E, C and combinations such as S+C.\n"
        f"active: {active}\n",
        encoding="utf-8",
    )


def safe_name(text: str) -> str:
    return text.replace("+", "p")


def empty_metrics() -> dict[str, float | None]:
    return {"precision": None, "recall": None, "map50": None, "map5095": None, "f1": None}


def metrics_from_mapping(values: dict) -> dict[str, float | None]:
    cleaned = {str(k).strip(): v for k, v in values.items()}

    def get_float(candidates: Iterable[str]) -> float | None:
        for key in candidates:
            if key in cleaned and str(cleaned[key]).strip():
                try:
                    return float(cleaned[key])
                except (TypeError, ValueError):
                    pass
        return None

    precision = get_float(("metrics/precision(M)", "metrics/precision(B)", "precision", "Precision"))
    recall = get_float(("metrics/recall(M)", "metrics/recall(B)", "recall", "Recall"))
    map50 = get_float(("metrics/mAP50(M)", "metrics/mAP50(B)", "mAP0.5", "map50"))
    map5095 = get_float(("metrics/mAP50-95(M)", "metrics/mAP50-95(B)", "mAP0.5-0.95", "map5095"))
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "map50": map50, "map5095": map5095, "f1": f1}


def latest_metrics(results_csv: Path) -> dict[str, float | None]:
    if not results_csv.exists():
        return empty_metrics()

    rows = list(csv.DictReader(results_csv.open("r", encoding="utf-8")))
    if not rows:
        return empty_metrics()
    return metrics_from_mapping(rows[-1])



def validation_log_metrics(log_path: Path) -> dict[str, float | None]:
    """Parse the final `all` row under the last Ultralytics `Validating` block.

    For segmentation logs, the `all` row contains box metrics followed by mask
    metrics. We use the final four values, i.e. mask P/R/mAP50/mAP50-95. For
    detection logs, the final four values are the detection metrics.
    """
    if not log_path.exists():
        return empty_metrics()
    ansi = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
    try:
        raw_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return empty_metrics()

    lines = [ansi.sub("", raw).replace("\r", "").strip() for raw in raw_lines]
    validating_idx = None
    for idx, line in enumerate(lines):
        if line.startswith("Validating ") or line.startswith("Validating/") or "Validating " in line:
            validating_idx = idx
    if validating_idx is None:
        return empty_metrics()

    for line in reversed(lines[validating_idx + 1 :]):
        if not line.startswith("all"):
            continue
        parts = line.split()
        if len(parts) < 7 or parts[0] != "all":
            continue
        nums = []
        for token in parts[1:]:
            try:
                nums.append(float(token))
            except ValueError:
                pass
        # Expected after class name: Images, Instances, then 4 or 8 metrics.
        if len(nums) >= 10:
            metrics = nums[-4:]
        elif len(nums) >= 6:
            metrics = nums[-4:]
        else:
            continue
        precision, recall, map50, map5095 = metrics
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else None
        return {"precision": precision, "recall": recall, "map50": map50, "map5095": map5095, "f1": f1}
    return empty_metrics()


def apply_metrics(record: RunRecord, metrics: dict[str, float | None]) -> RunRecord:
    if metrics.get("map5095") is None:
        return record
    record.precision = metrics["precision"]
    record.recall = metrics["recall"]
    record.f1 = metrics["f1"]
    record.map50 = metrics["map50"]
    record.map5095 = metrics["map5095"]
    return record


def checkpoint_metrics(checkpoint: Path, args) -> dict[str, float | None]:
    if not checkpoint.exists():
        return empty_metrics()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        val_model = YOLO(str(checkpoint))
        results = val_model.val(
            data=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            conf=None,
            iou=0.7,
            augment=False,
            rect=False,
            split="val",
            plots=False,
            verbose=False,
        )
        return metrics_from_mapping(getattr(results, "results_dict", {}) or {})
    except Exception as exc:  # fallback keeps long search runs alive if val has a transient issue
        print(f"[WARN] best checkpoint validation failed for {checkpoint}: {exc}", file=sys.stderr)
        return empty_metrics()


def train_once(
    args,
    subset: str,
    active: str,
    epochs: int,
    name: str,
    model_source: str | Path,
    resume: bool = False,
    stop_epoch: int | None = None,
) -> RunRecord:
    config_path = Path(args.config)
    write_moe_config(active, config_path)
    os.environ["ES_MOE_CONFIG"] = str(config_path.resolve())

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = YOLO(str(model_source))
    if hasattr(model, "float"):
        model = model.float()

    stage_stop_epoch = stop_epoch if stop_epoch and 0 < stop_epoch < epochs and not resume else None
    if stage_stop_epoch is not None:
        def _stop_at_epoch(trainer):
            current_epoch = int(getattr(trainer, "epoch", -1)) + 1
            if current_epoch >= stage_stop_epoch:
                print(
                    f"[STOP_EPOCH] reached {current_epoch}/{epochs}; "
                    f"stopping this stage but keeping full_epochs={epochs} in args",
                    flush=True,
                )
                trainer.stop = True

        model.add_callback("on_train_epoch_end", _stop_at_epoch)

    start = time.perf_counter()
    train_kwargs = dict(
        data=args.data,
        epochs=epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
    )
    if resume:
        train_kwargs["resume"] = True
    else:
        train_kwargs.update(project=args.project, name=name, exist_ok=args.exist_ok)
    model.train(**train_kwargs)
    hours = (time.perf_counter() - start) / 3600.0

    save_dir = Path(model.trainer.save_dir) if getattr(model, "trainer", None) else Path(args.project) / name
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"
    # Use the trainer-produced validation record. Do not run an extra YOLO(best.pt).val() here.
    metrics = latest_metrics(save_dir / "results.csv")
    gpu_gb = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0

    completed_epochs = stage_stop_epoch or epochs

    return RunRecord(
        subset=subset,
        active=active,
        epochs=completed_epochs,
        hours=hours,
        estimated_full_hours=hours * args.full_epochs / max(completed_epochs, 1),
        gpu_gb=gpu_gb,
        save_dir=str(save_dir),
        best=str(best) if best.exists() else "",
        last=str(last) if last.exists() else "",
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        map50=metrics["map50"],
        map5095=metrics["map5095"],
    )


def record_to_dict(r: RunRecord) -> dict:
    return {
        "subset": r.subset,
        "active": r.active,
        "epochs": r.epochs,
        "hours": r.hours,
        "estimated_full_hours": r.estimated_full_hours,
        "gpu_gb": r.gpu_gb,
        "precision": r.precision,
        "recall": r.recall,
        "f1": r.f1,
        "mAP0.5": r.map50,
        "mAP0.5-0.95": r.map5095,
        "rank_score": r.rank_score,
        "save_dir": r.save_dir,
        "best": r.best,
        "last": r.last,
    }


def record_from_dict(row: dict) -> RunRecord:
    def num(key: str) -> float | None:
        value = row.get(key, "")
        return float(value) if value not in ("", "None", None) else None

    return RunRecord(
        subset=row["subset"],
        active=row.get("active", row["subset"]),
        epochs=int(float(row["epochs"])),
        hours=float(row["hours"]),
        estimated_full_hours=float(row["estimated_full_hours"]),
        gpu_gb=float(row.get("gpu_gb", 0) or 0),
        save_dir=row.get("save_dir", ""),
        best=row.get("best", ""),
        last=row.get("last", ""),
        precision=num("precision"),
        recall=num("recall"),
        f1=num("f1"),
        map50=num("mAP0.5"),
        map5095=num("mAP0.5-0.95"),
        rank_score=num("rank_score"),
    )


def write_csv(path: Path, records: list[RunRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(record_to_dict(records[0]).keys()) if records else ["subset"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow(record_to_dict(r))


def load_profile_csv(path: Path) -> list[RunRecord]:
    with path.open("r", encoding="utf-8") as f:
        return [record_from_dict(row) for row in csv.DictReader(f)]


def gpu_total_gb() -> float:
    if not torch.cuda.is_available():
        return 24.0
    device = 0
    if isinstance(args_global.device, str) and args_global.device not in {"", "cpu"}:
        first = args_global.device.split(",")[0]
        if first.isdigit():
            device = int(first)
    return torch.cuda.get_device_properties(device).total_memory / 1024**3


def job_mem_gb(job: TrainJob, profile_by_subset: dict[str, RunRecord] | None, args) -> float:
    if profile_by_subset and job.subset in profile_by_subset and profile_by_subset[job.subset].gpu_gb > 0:
        return profile_by_subset[job.subset].gpu_gb
    return args.default_job_gpu_gb


def make_worker_command(args, job: TrainJob, tag: str) -> tuple[list[str], Path, Path]:
    worker_root = Path(args.outdir) / "workers"
    cfg_path = worker_root / "configs" / f"{tag}_{safe_name(job.subset)}.yaml"
    out_path = worker_root / "records" / f"{tag}_{safe_name(job.subset)}.json"
    log_path = worker_root / "logs" / f"{tag}_{safe_name(job.subset)}.log"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode",
        "worker",
        "--worker-subset",
        job.subset,
        "--worker-active",
        job.active,
        "--worker-epochs",
        str(job.epochs),
        "--worker-stop-epoch",
        str(job.stop_epoch or 0),
        "--worker-name",
        job.name,
        "--worker-model-source",
        str(job.model_source),
        "--worker-output",
        str(out_path),
        "--config",
        str(cfg_path),
        "--model",
        args.model,
        "--data",
        args.data,
        "--project",
        args.project,
        "--outdir",
        args.outdir,
        "--full-epochs",
        str(args.full_epochs),
        "--profile-epochs",
        str(args.profile_epochs),
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--device",
        str(args.device),
        "--workers",
        str(args.workers),
        "--patience",
        str(args.patience),
    ]
    if args.exist_ok:
        cmd.append("--exist-ok")
    if job.resume:
        cmd.append("--worker-resume")
    return cmd, out_path, log_path


def launch_batch(args, jobs: list[TrainJob], tag: str) -> list[RunRecord]:
    processes = []
    for job in jobs:
        cmd, out_path, log_path = make_worker_command(args, job, tag)
        log_f = log_path.open("w", encoding="utf-8")
        stop_text = f", stop_epoch={job.stop_epoch}" if job.stop_epoch else ""
        print(f"[PARALLEL] start {job.subset}: epochs={job.epochs}{stop_text}, resume={job.resume}, log={log_path}")
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        processes.append((job, proc, log_f, out_path, log_path))

    records: list[RunRecord] = []
    for job, proc, log_f, out_path, log_path in processes:
        code = proc.wait()
        log_f.close()
        if code != 0:
            tail = ""
            try:
                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                tail = "\n".join(lines[-60:])
            except Exception:
                pass
            raise RuntimeError(f"Worker failed for {job.subset} with code {code}. Log: {log_path}\n{tail}")
        rec = record_from_dict(json.loads(out_path.read_text(encoding="utf-8")))
        log_metrics = validation_log_metrics(log_path)
        rec = apply_metrics(rec, log_metrics)
        records.append(rec)
        metric_text = ""
        if log_metrics.get("map5095") is not None:
            metric_text = f" P={rec.precision:.5f} R={rec.recall:.5f} mAP50={rec.map50:.5f} mAP50-95={rec.map5095:.5f}"
        print(f"[PARALLEL] done {job.subset}{metric_text}")
    return records


def build_memory_batches(args, jobs: list[TrainJob], profile: list[RunRecord] | None) -> list[list[TrainJob]]:
    if args.parallel <= 1:
        return [[job] for job in jobs]
    profile_by_subset = {r.subset: r for r in profile} if profile else None
    total = gpu_total_gb()
    max_gb = total * args.gpu_max_fraction
    target_gb = total * args.gpu_target_fraction

    remaining = sorted(jobs, key=lambda j: job_mem_gb(j, profile_by_subset, args), reverse=True)
    batches: list[list[TrainJob]] = []
    while remaining:
        batch: list[TrainJob] = []
        batch_mem = 0.0
        for job in list(remaining):
            mem = job_mem_gb(job, profile_by_subset, args)
            if mem > max_gb:
                raise RuntimeError(
                    f"Stop to avoid likely OOM: job {job.subset} estimates {mem:.2f}GB, "
                    f"above max budget {max_gb:.2f}GB. Lower --batch or increase --gpu-max-fraction."
                )
            if len(batch) < args.parallel and batch_mem + mem <= max_gb:
                batch.append(job)
                batch_mem += mem
                remaining.remove(job)
            if len(batch) >= args.parallel or batch_mem >= target_gb:
                break
        if not batch:
            next_job = remaining[0]
            mem = job_mem_gb(next_job, profile_by_subset, args)
            raise RuntimeError(
                f"Stop to avoid likely OOM: cannot schedule {next_job.subset} "
                f"without exceeding max budget {max_gb:.2f}GB. Estimated job memory: {mem:.2f}GB."
            )
        labels = ", ".join(j.subset for j in batch)
        print(f"[SCHEDULE] batch {len(batches) + 1}: {labels}; estimated GPU={batch_mem:.2f}GB / target={target_gb:.2f}GB / max={max_gb:.2f}GB")
        batches.append(batch)
    return batches


def run_training_jobs(args, jobs: list[TrainJob], tag: str, profile: list[RunRecord] | None = None) -> list[RunRecord]:
    records: list[RunRecord] = []
    if args.parallel <= 1:
        # Even serial runs use a one-job worker subprocess so that we can parse
        # the exact Ultralytics built-in validation line from its saved log.
        for idx, job in enumerate(jobs, 1):
            records.extend(launch_batch(args, [job], f"{tag}_s{idx}"))
        return records

    for idx, batch in enumerate(build_memory_batches(args, jobs, profile), 1):
        records.extend(launch_batch(args, batch, f"{tag}_b{idx}"))
    return records


def run_profile(args) -> list[RunRecord]:
    jobs = [
        TrainJob(
            subset=subset,
            active=active,
            epochs=args.profile_epochs,
            name=f"profile_{safe_name(subset)}_{args.profile_epochs}ep",
            model_source=args.model,
        )
        for subset, active in SUBSETS.items()
    ]
    records = run_training_jobs(args, jobs, tag=f"profile_{args.profile_epochs}ep", profile=None)

    profile_csv = Path(args.outdir) / f"profile_{args.profile_epochs}epoch.csv"
    write_csv(profile_csv, records)
    (Path(args.outdir) / f"profile_{args.profile_epochs}epoch.json").write_text(
        json.dumps([record_to_dict(r) for r in records], indent=2), encoding="utf-8"
    )
    print(f"\nSaved profiling table: {profile_csv}")
    return records


def _metric_value(value: float | None) -> float:
    return float(value) if value is not None and not math.isnan(value) else 0.0


def tolerant_ranks(values: dict[str, float | None], tolerance: float, reverse: bool = True) -> dict[str, int]:
    valid = [(k, v) for k, v in values.items() if v is not None and not math.isnan(v)]
    ordered = sorted(valid, key=lambda kv: kv[1], reverse=reverse)
    result = {k: len(values) + 1 for k in values}
    if not ordered:
        return result

    current_rank = 1
    anchor_value = ordered[0][1]
    result[ordered[0][0]] = current_rank
    for position, (key, value) in enumerate(ordered[1:], 2):
        if abs(value - anchor_value) >= tolerance:
            current_rank = position
            anchor_value = value
        result[key] = current_rank
    return result


def _margin_compare(a: RunRecord, b: RunRecord) -> int:
    """Tie-break equal rank scores by accumulated absolute metric advantages."""
    metrics = ("f1", "map50", "map5095")
    a_advantage = 0.0
    b_advantage = 0.0
    for metric in metrics:
        delta = _metric_value(getattr(a, metric)) - _metric_value(getattr(b, metric))
        if delta > 0:
            a_advantage += delta
        elif delta < 0:
            b_advantage += -delta

    if a_advantage > b_advantage:
        return -1
    if b_advantage > a_advantage:
        return 1

    # Final tie-break by the agreed metric importance: mAP50-95 > mAP50 > F1.
    for metric in ("map5095", "map50", "f1"):
        delta = _metric_value(getattr(a, metric)) - _metric_value(getattr(b, metric))
        if delta > 0:
            return -1
        if delta < 0:
            return 1
    return 0


def rank_records(records: list[RunRecord], args=None) -> list[RunRecord]:
    tolerance = getattr(args, "rank_tolerance", 0.003) if args is not None else 0.003
    f1r = tolerant_ranks({r.subset: r.f1 for r in records}, tolerance, reverse=True)
    m50r = tolerant_ranks({r.subset: r.map50 for r in records}, tolerance, reverse=True)
    m95r = tolerant_ranks({r.subset: r.map5095 for r in records}, tolerance, reverse=True)

    for r in records:
        r.rank_score = f1r[r.subset] + m50r[r.subset] + m95r[r.subset]

    def compare(a: RunRecord, b: RunRecord) -> int:
        a_score = a.rank_score if a.rank_score is not None else 999
        b_score = b.rank_score if b.rank_score is not None else 999
        if a_score < b_score:
            return -1
        if a_score > b_score:
            return 1
        return _margin_compare(a, b)

    return sorted(records, key=cmp_to_key(compare))


def normalize_saving(value: float) -> float:
    return value / 100.0 if value > 1 else value


def epoch_cost_plan(
    num_subsets: int,
    full_epochs: int,
    profile_epochs: int,
    saving: float,
    keep: int | None = None,
    initial_epochs: int | None = None,
    min_initial_epochs: int = 1,
) -> BudgetPlan:
    if num_subsets < 2:
        raise ValueError("Need at least two subsets for staged selection.")
    if not 0 < profile_epochs < full_epochs:
        raise ValueError("profile_epochs must be between 1 and full_epochs - 1.")
    target = normalize_saving(saving)
    if not 0 < target < 1:
        raise ValueError("saving must be in (0, 1) or percent in (0, 100).")

    full_cost = num_subsets * full_epochs
    target_cost = (1.0 - target) * full_cost

    if keep is not None and not 1 <= keep < num_subsets:
        raise ValueError(f"keep must be between 1 and {num_subsets - 1}.")
    if initial_epochs is not None and not profile_epochs <= initial_epochs < full_epochs:
        raise ValueError("initial_epochs must be between profile_epochs and full_epochs - 1.")

    if keep is not None and initial_epochs is None:
        initial_epochs = math.floor((target_cost - num_subsets * profile_epochs - keep * full_epochs) / (num_subsets - keep))
    elif initial_epochs is not None and keep is None:
        keep = math.floor((target_cost - num_subsets * profile_epochs - num_subsets * initial_epochs) / (full_epochs - initial_epochs))
    elif keep is None and initial_epochs is None:
        candidates: list[tuple[int, int, int, float]] = []
        for k in range(1, num_subsets):
            e = math.floor((target_cost - num_subsets * profile_epochs - k * full_epochs) / (num_subsets - k))
            if profile_epochs <= e < full_epochs and e >= min_initial_epochs:
                cost = num_subsets * profile_epochs + num_subsets * e + k * (full_epochs - e)
                actual_saving = 1.0 - cost / full_cost
                candidates.append((k, e, cost, actual_saving))
        if not candidates:
            raise ValueError("No feasible plan found. Try a smaller saving, fewer profile epochs, or manually set --keep/--initial-epochs.")
        keep, initial_epochs, _, _ = max(candidates, key=lambda x: (x[0], x[1]))
    else:
        assert keep is not None and initial_epochs is not None

    if keep is None or initial_epochs is None:
        raise RuntimeError("Failed to derive keep and initial_epochs.")
    if initial_epochs < profile_epochs:
        raise ValueError(f"Derived initial_epochs={initial_epochs}, smaller than profile_epochs={profile_epochs}. Use smaller --keep or lower --saving.")
    if keep < 1 or keep >= num_subsets:
        raise ValueError(f"Derived keep={keep}, outside valid range 1..{num_subsets - 1}.")

    epoch_cost = num_subsets * profile_epochs + num_subsets * initial_epochs + keep * (full_epochs - initial_epochs)
    actual_saving = 1.0 - epoch_cost / full_cost
    return BudgetPlan(
        saving_percent=target * 100.0,
        initial_epochs=int(initial_epochs),
        keep=int(keep),
        epoch_cost=int(epoch_cost),
        full_epoch_cost=int(full_cost),
        actual_saving_percent=actual_saving * 100.0,
    )


def plan_time_cost(profile: list[RunRecord], plan: BudgetPlan, selected: Iterable[str] | None = None) -> dict:
    full = sum(r.estimated_full_hours for r in profile)
    profile_cost = sum(r.hours for r in profile)
    screen_cost = sum(r.estimated_full_hours * plan.initial_epochs / args_global.full_epochs for r in profile)
    remaining_ratio = (args_global.full_epochs - plan.initial_epochs) / args_global.full_epochs

    by_subset = {r.subset: r for r in profile}
    if selected is not None:
        continuation = sum(by_subset[s].estimated_full_hours * remaining_ratio for s in selected)
        total = profile_cost + screen_cost + continuation
        return {"total": total, "saving": 1 - total / full, "full": full}

    continuation_costs = [r.estimated_full_hours * remaining_ratio for r in profile]
    best_cont = sum(sorted(continuation_costs)[: plan.keep])
    worst_cont = sum(sorted(continuation_costs, reverse=True)[: plan.keep])
    best_total = profile_cost + screen_cost + best_cont
    conservative_total = profile_cost + screen_cost + worst_cont
    return {
        "best_total": best_total,
        "conservative_total": conservative_total,
        "best_saving": 1 - best_total / full,
        "conservative_saving": 1 - conservative_total / full,
        "full": full,
        # Backward-compatible aliases for older summaries/scripts.
        "min_total": best_total,
        "max_total": conservative_total,
    }


def fixed_strategy_plan(
    num_subsets: int,
    full_epochs: int,
    profile_epochs: int,
    keep: int,
    initial_epochs: int,
) -> BudgetPlan:
    if not 1 <= keep < num_subsets:
        raise ValueError(f"--keep must be between 1 and {num_subsets - 1}.")
    if not profile_epochs <= initial_epochs < full_epochs:
        raise ValueError("--initial-epochs must be between profile_epochs and full_epochs - 1.")
    full_cost = num_subsets * full_epochs
    epoch_cost = num_subsets * profile_epochs + num_subsets * initial_epochs + keep * (full_epochs - initial_epochs)
    actual_saving = 1.0 - epoch_cost / full_cost
    return BudgetPlan(
        saving_percent=actual_saving * 100.0,
        initial_epochs=int(initial_epochs),
        keep=int(keep),
        epoch_cost=int(epoch_cost),
        full_epoch_cost=int(full_cost),
        actual_saving_percent=actual_saving * 100.0,
    )


def derive_requested_plans(args) -> list[BudgetPlan]:
    if args.keep is None or args.initial_epochs is None:
        raise ValueError("--mode plan/run requires --keep K and --initial-epochs E1. Saving is computed as an output.")
    return [
        fixed_strategy_plan(
            num_subsets=len(SUBSETS),
            full_epochs=args.full_epochs,
            profile_epochs=args.profile_epochs,
            keep=args.keep,
            initial_epochs=args.initial_epochs,
        )
    ]


def print_policy_plans(profile: list[RunRecord], plans: list[BudgetPlan], args) -> None:
    full_time = sum(r.estimated_full_hours for r in profile)
    profile_time = sum(r.hours for r in profile)
    print("\nEstimated full independent training time")
    print("subset,profile_h,estimated_full_h,gpu_gb")
    for r in profile:
        print(f"{r.subset},{r.hours:.3f},{r.estimated_full_hours:.3f},{r.gpu_gb:.2f}")
    print(f"Total full time estimate: {full_time:.3f} h")
    print(f"Profiling cost ({args.profile_epochs} epochs x {len(SUBSETS)} subsets): {profile_time:.3f} h")

    print("\nGeneric epoch-cost plan")
    print("actual_saving,full_epochs,profile_epochs,initial_epochs,keep,epoch_cost")
    for plan in plans:
        print(
            f"{plan.actual_saving_percent:.2f}%,{args.full_epochs},{args.profile_epochs},"
            f"{plan.initial_epochs},{plan.keep},{plan.epoch_cost}/{plan.full_epoch_cost}"
        )

    print("\nConservative accumulated completed-hours estimate from profiling")
    print("actual_epoch_saving,initial_epochs,keep,conservative_completed_hours,conservative_completed_hours_saving")
    for plan in plans:
        cost = plan_time_cost(profile, plan)
        print(
            f"{plan.saving_percent:.2f}%,{plan.initial_epochs},{plan.keep},"
            f"{cost['conservative_total']:.3f},"
            f"{cost['conservative_saving']*100:.2f}%"
        )




def snapshot_stage_last(records: list[RunRecord], tag: str) -> None:
    """Copy stage1 last.pt to an immutable snapshot before Stage2 resume."""
    for r in records:
        last = Path(r.last)
        if not last.exists():
            continue
        snapshot = last.with_name(f"last_{tag}_snapshot.pt")
        if not snapshot.exists():
            shutil.copy2(last, snapshot)
        r.last = str(snapshot)


def load_worker_records_from_logs(args, tag_prefix: str) -> list[RunRecord]:
    records_dir = Path(args.outdir) / "workers" / "records"
    logs_dir = Path(args.outdir) / "workers" / "logs"
    records: list[RunRecord] = []
    for record_path in sorted(records_dir.glob(f"{tag_prefix}_*.json")):
        rec = record_from_dict(json.loads(record_path.read_text(encoding="utf-8")))
        log_path = logs_dir / f"{record_path.stem}.log"
        rec = apply_metrics(rec, validation_log_metrics(log_path))
        records.append(rec)
    return records


def latest_record_for_subset(records: list[RunRecord], subset: str) -> RunRecord | None:
    matches = [r for r in records if r.subset == subset]
    if not matches:
        return None
    return sorted(matches, key=lambda r: (r.epochs, r.hours))[-1]


def run_continue_stage2(args) -> None:
    if args.keep is None or args.initial_epochs is None:
        raise ValueError("--mode continue-stage2 requires --keep K and --initial-epochs E.")

    plan = fixed_strategy_plan(
        num_subsets=len(SUBSETS),
        full_epochs=args.full_epochs,
        profile_epochs=args.profile_epochs,
        keep=args.keep,
        initial_epochs=args.initial_epochs,
    )
    tag = plan.tag
    stage1_prefix = f"budget{tag}_stage1"
    final_prefix = f"budget{tag}_final"

    stage_records = load_worker_records_from_logs(args, stage1_prefix)
    if len(stage_records) < len(SUBSETS):
        raise FileNotFoundError(
            f"Expected at least {len(SUBSETS)} stage1 records for prefix {stage1_prefix}, found {len(stage_records)}."
        )
    # Keep only one record per subset if there are duplicate worker attempts.
    by_subset = {subset: latest_record_for_subset(stage_records, subset) for subset in SUBSETS}
    missing = [k for k, v in by_subset.items() if v is None]
    if missing:
        raise FileNotFoundError(f"Missing stage1 records for subsets: {missing}")
    stage_records = [by_subset[subset] for subset in SUBSETS if by_subset[subset] is not None]
    snapshot_stage_last(stage_records, f"stage1_{tag}")

    ranked = rank_records(stage_records, args)
    selected = [r.subset for r in ranked[: args.keep]]
    write_csv(Path(args.outdir) / f"policy_{tag}_stage1_ranked.csv", ranked)
    print(f"[CONTINUE] Re-ranked Stage1 from Validating logs. Top-{args.keep}: {', '.join(selected)}")

    existing_final = load_worker_records_from_logs(args, final_prefix)
    existing_by_subset = {r.subset: latest_record_for_subset(existing_final, r.subset) for r in existing_final}
    if all(subset in existing_by_subset and existing_by_subset[subset] is not None for subset in selected):
        final_records = [existing_by_subset[subset] for subset in selected]
        ranked_final = rank_records(final_records, args)
        write_csv(Path(args.outdir) / f"policy_{tag}_final_ranked.csv", ranked_final)
        print(f"[CONTINUE] Existing Stage2 logs found. Re-ranked final from Validating logs: {Path(args.outdir) / f'policy_{tag}_final_ranked.csv'}")
        return

    print("[CONTINUE] Stage2 logs are incomplete; launching resume jobs for selected subsets.")
    final_jobs: list[TrainJob] = []
    for subset in selected:
        source = by_subset[subset].last if by_subset[subset] is not None else ""
        if not source:
            raise FileNotFoundError(f"No stage1 last.pt found for subset {subset}; cannot resume.")
        final_jobs.append(
            TrainJob(
                subset=subset,
                active=SUBSETS[subset],
                epochs=args.full_epochs,
                name=f"budget{tag}_final_{safe_name(subset)}_{args.full_epochs}ep",
                model_source=source,
                resume=True,
            )
        )
    final_records = run_training_jobs(args, final_jobs, tag=final_prefix, profile=None)
    ranked_final = rank_records(final_records, args)
    write_csv(Path(args.outdir) / f"policy_{tag}_final_ranked.csv", ranked_final)
    print(f"[CONTINUE] Stage2 complete. Final ranking saved: {Path(args.outdir) / f'policy_{tag}_final_ranked.csv'}")


def run_policy(args, profile: list[RunRecord], plan: BudgetPlan) -> None:
    initial_epochs = plan.initial_epochs
    keep = plan.keep
    tag = plan.tag

    print(
        f"\n[PLAN k={keep},e={initial_epochs}] Stage 1: train all subsets with "
        f"full_epochs={args.full_epochs}, then stop at epoch {initial_epochs}"
    )
    stage_jobs = [
        TrainJob(
            subset=subset,
            active=active,
            epochs=args.full_epochs,
            name=f"budget{tag}_stage1_{safe_name(subset)}_stop{initial_epochs}of{args.full_epochs}ep",
            model_source=args.model,
            stop_epoch=initial_epochs,
        )
        for subset, active in SUBSETS.items()
    ]
    stage_records = run_training_jobs(args, stage_jobs, tag=f"budget{tag}_stage1", profile=profile)
    snapshot_stage_last(stage_records, f"stage1_{tag}")

    ranked = rank_records(stage_records, args)
    selected = [r.subset for r in ranked[:keep]]
    print(f"\n[PLAN k={keep},e={initial_epochs}] Selected Top-{keep}: {', '.join(selected)}")
    write_csv(Path(args.outdir) / f"policy_{tag}_stage1_ranked.csv", ranked)

    if args.stop_after_stage1:
        actual = plan_time_cost(profile, plan, selected)
        report = {
            "saving_percent": plan.saving_percent,
            "full_epochs": args.full_epochs,
            "profile_epochs": args.profile_epochs,
            "initial_epochs": plan.initial_epochs,
            "keep": plan.keep,
            "parallel": args.parallel,
            "stopped_after_stage1": True,
            "selected": selected,
            "actual_profile_completed_hours": sum(r.hours for r in profile),
            "actual_stage1_completed_hours": sum(r.hours for r in stage_records),
            "actual_accumulated_completed_hours_so_far": sum(r.hours for r in profile) + sum(r.hours for r in stage_records),
            "estimated_accumulated_completed_hours_if_resumed": actual["total"],
            "estimated_accumulated_completed_saving_if_resumed": actual["saving"],
        }
        (Path(args.outdir) / f"policy_{tag}_stage1_only_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n[PLAN k={keep},e={initial_epochs}] stop-after-stage1 enabled; stage2 resume is skipped.")
        print(f"[PLAN k={keep},e={initial_epochs}] Stage1 completed-hours: {sum(r.hours for r in stage_records):.3f} h")
        return

    remaining = args.full_epochs - initial_epochs
    final_records: list[RunRecord] = []
    if remaining > 0:
        print(
            f"\n[PLAN k={keep},e={initial_epochs}] Stage 2: true resume selected subsets "
            f"from {initial_epochs} to {args.full_epochs} total epochs"
        )
        by_subset = {r.subset: r for r in stage_records}
        final_jobs: list[TrainJob] = []
        for subset in selected:
            source = by_subset[subset].last
            if not source:
                raise FileNotFoundError(f"No last.pt found for subset {subset}; cannot resume strictly.")
            final_jobs.append(
                TrainJob(
                    subset=subset,
                    active=SUBSETS[subset],
                    epochs=args.full_epochs,
                    name=f"budget{tag}_final_{safe_name(subset)}_{args.full_epochs}ep",
                    model_source=source,
                    resume=True,
                )
            )
        final_records = run_training_jobs(args, final_jobs, tag=f"budget{tag}_final", profile=profile)

    if final_records:
        ranked_final = rank_records(final_records, args)
        write_csv(Path(args.outdir) / f"policy_{tag}_final_ranked.csv", ranked_final)

    actual = plan_time_cost(profile, plan, selected)
    report = {
        "saving_percent": plan.saving_percent,
        "full_epochs": args.full_epochs,
        "profile_epochs": args.profile_epochs,
        "initial_epochs": plan.initial_epochs,
        "keep": plan.keep,
        "parallel": args.parallel,
        "gpu_target_fraction": args.gpu_target_fraction,
        "gpu_max_fraction": args.gpu_max_fraction,
        "epoch_cost": plan.epoch_cost,
        "full_epoch_cost": plan.full_epoch_cost,
        "actual_epoch_saving_percent": plan.actual_saving_percent,
        "selected": selected,
        "estimated_accumulated_completed_hours_from_profile": actual["total"],
        "estimated_accumulated_completed_saving_from_profile": actual["saving"],
        "actual_profile_completed_hours": sum(r.hours for r in profile),
        "actual_stage1_completed_hours": sum(r.hours for r in stage_records),
        "actual_final_completed_hours": sum(r.hours for r in final_records),
        "actual_accumulated_completed_hours": (
            sum(r.hours for r in profile) + sum(r.hours for r in stage_records) + sum(r.hours for r in final_records)
        ),
        "actual_accumulated_completed_saving_vs_profile_full_estimate": 1
        - (sum(r.hours for r in profile) + sum(r.hours for r in stage_records) + sum(r.hours for r in final_records))
        / actual["full"],
    }
    (Path(args.outdir) / f"policy_{tag}_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    actual_completed = (
        sum(r.hours for r in profile) + sum(r.hours for r in stage_records) + sum(r.hours for r in final_records)
    )
    actual_saving = 1 - actual_completed / actual["full"]
    print(
        f"\n[PLAN k={keep},e={initial_epochs}] Estimated accumulated completed-hours from profiling: "
        f"{actual['total']:.3f} h; saving: {actual['saving'] * 100:.2f}%"
    )
    print(
        f"[PLAN k={keep},e={initial_epochs}] Actual accumulated completed-hours from finished runs: "
        f"{actual_completed:.3f} h; saving vs profile full estimate: {actual_saving * 100:.2f}%"
    )



def run_worker(args) -> None:
    rec = train_once(
        args,
        subset=args.worker_subset,
        active=args.worker_active,
        epochs=args.worker_epochs,
        name=args.worker_name,
        model_source=args.worker_model_source,
        resume=args.worker_resume,
        stop_epoch=args.worker_stop_epoch or None,
    )
    output = Path(args.worker_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record_to_dict(rec), indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile and run generic budgeted ES_MOE expert subset search.")
    parser.add_argument("--mode", choices=("profile", "plan", "run", "continue-stage2", "worker"), default="profile")
    parser.add_argument("--keep", type=int, default=None, help="Retained subset count K. Required for plan/run.")
    parser.add_argument("--initial-epochs", type=int, default=None, help="Stage1 stop epoch e. Stage1 still trains with --full-epochs but stops after this epoch.")
    parser.add_argument("--min-initial-epochs", type=int, default=1, help="Minimum screening epoch for auto plan search.")
    parser.add_argument("--parallel", type=int, default=3, help="Maximum concurrent training processes. Use 1 for serial.")
    parser.add_argument("--stop-after-stage1", action="store_true", help="Run profiling and stage1 ranking only, then stop before final resume.")
    parser.add_argument("--rank-tolerance", type=float, default=0.003, help="Treat metric differences below this value as tied when computing primary rank scores.")
    parser.add_argument("--gpu-target-fraction", type=float, default=0.67, help="Try to pack each parallel batch to at least this fraction of GPU memory when possible.")
    parser.add_argument("--gpu-max-fraction", type=float, default=0.90, help="Do not schedule a parallel batch above this estimated GPU memory fraction.")
    parser.add_argument("--default-job-gpu-gb", type=float, default=7.0, help="Fallback GPU GB estimate before profiling is available.")
    parser.add_argument("--model", default="ultralytics/cfg/models/master/v0/seg/yolo-master-seg-n.yaml")
    parser.add_argument("--data", default="/root/lanyun-tmp/YOLO-Master/26944/26922/datasets/shipseg_coco/data.yaml")
    parser.add_argument("--config", default="moe_expert_config.yaml")
    parser.add_argument("--project", default="runs/expert_budget_search")
    parser.add_argument("--outdir", default="runs/expert_budget_search/reports")
    parser.add_argument("--profile-csv", default="")
    parser.add_argument("--full-epochs", type=int, default=200)
    parser.add_argument("--profile-epochs", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--worker-subset", default="")
    parser.add_argument("--worker-active", default="")
    parser.add_argument("--worker-epochs", type=int, default=0)
    parser.add_argument("--worker-stop-epoch", type=int, default=0)
    parser.add_argument("--worker-name", default="")
    parser.add_argument("--worker-model-source", default="")
    parser.add_argument("--worker-output", default="")
    parser.add_argument("--worker-resume", action="store_true")
    return parser.parse_args()


def load_or_run_profile(args) -> list[RunRecord]:
    if args.profile_csv:
        return load_profile_csv(Path(args.profile_csv))
    default_csv = Path(args.outdir) / f"profile_{args.profile_epochs}epoch.csv"
    legacy_csv = Path(args.outdir) / "profile_5epoch.csv"
    if args.mode in {"plan", "run"}:
        if default_csv.exists():
            return load_profile_csv(default_csv)
        if args.profile_epochs == 5 and legacy_csv.exists():
            return load_profile_csv(legacy_csv)
    return run_profile(args)


args_global: argparse.Namespace


def main() -> None:
    global args_global
    args = parse_args()
    args_global = args
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    if args.mode == "worker":
        run_worker(args)
        return

    if args.mode == "profile":
        profile = run_profile(args)
        print("\nProfile completed. Use --mode plan/run with --keep and --initial-epochs to compute savings as outputs.")
        return

    if args.mode == "continue-stage2":
        run_continue_stage2(args)
        return

    plans = derive_requested_plans(args)
    profile = load_or_run_profile(args)
    print_policy_plans(profile, plans, args)

    if args.mode == "run":
        run_policy(args, profile, plans[0])


if __name__ == "__main__":
    main()
