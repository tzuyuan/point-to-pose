import argparse
import csv
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from glob import glob
from pathlib import Path


AVG_LINE_RE = re.compile(
    r"Average,\s+ADD-S_err:\s*([0-9.+-eE]+)\[cm\],\s*"
    r"ADD_errs:\s*([0-9.+-eE]+)\[cm\],\s*"
    r"ADD-S_AUC:\s*([0-9.+-eE]+),\s*"
    r"ADD_AUC:\s*([0-9.+-eE]+),\s*"
    r"mesh_CD:\s*([0-9.+-eE]+)\[cm\]"
)


def _find_configs(config_glob: str) -> list[Path]:
    configs = [Path(p).resolve() for p in glob(config_glob)]
    return sorted(configs, key=lambda p: p.name)


def _parse_average_metrics(summary_path: Path) -> dict:
    metrics = {
        "avg_add_s_err_cm": "",
        "avg_add_err_cm": "",
        "avg_add_s_auc": "",
        "avg_add_auc": "",
        "avg_mesh_cd_cm": "",
    }
    if not summary_path.exists():
        return metrics

    try:
        content = summary_path.read_text(encoding="utf-8")
    except Exception:
        return metrics

    for line in content.splitlines():
        m = AVG_LINE_RE.search(line)
        if not m:
            continue
        metrics["avg_add_s_err_cm"] = m.group(1)
        metrics["avg_add_err_cm"] = m.group(2)
        metrics["avg_add_s_auc"] = m.group(3)
        metrics["avg_add_auc"] = m.group(4)
        metrics["avg_mesh_cd_cm"] = m.group(5)
        break
    return metrics


def _run_single_config(
    python_exe: str,
    runner_script: Path,
    data_path: Path,
    config_path: Path,
    out_root: Path,
) -> dict:
    config_name = config_path.stem
    run_out_dir = out_root / config_name
    run_out_dir.mkdir(parents=True, exist_ok=True)

    config_copy = run_out_dir / "config_used.yaml"
    shutil.copy2(config_path, config_copy)

    log_path = run_out_dir / "ablation_run.log"
    cmd = [
        python_exe,
        str(runner_script),
        "--data_path",
        str(data_path),
        "--out_dir",
        str(run_out_dir),
        "--config_path",
        str(config_path),
    ]

    start_time = time.time()
    start_iso = datetime.now().isoformat(timespec="seconds")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Start: {start_iso}\n")
        f.write(f"Command: {' '.join(cmd)}\n\n")
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
    elapsed_s = time.time() - start_time

    summary_path = run_out_dir / "summary_results.txt"
    metrics = _parse_average_metrics(summary_path)

    return {
        "config_name": config_name,
        "config_path": str(config_path),
        "status": "ok" if proc.returncode == 0 else "failed",
        "return_code": proc.returncode,
        "elapsed_s": f"{elapsed_s:.2f}",
        "run_out_dir": str(run_out_dir),
        "log_path": str(log_path),
        "summary_path": str(summary_path) if summary_path.exists() else "",
        **metrics,
    }


def _write_summary(out_root: Path, rows: list[dict]) -> None:
    summary_csv = out_root / "ablation_runs_summary.csv"
    fieldnames = [
        "config_name",
        "status",
        "return_code",
        "elapsed_s",
        "config_path",
        "run_out_dir",
        "log_path",
        "summary_path",
        "avg_add_s_err_cm",
        "avg_add_err_cm",
        "avg_add_s_auc",
        "avg_add_auc",
        "avg_mesh_cd_cm",
    ]
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nSaved ablation summary CSV: {summary_csv}")


def run_ho3d_ablation(
    data_path: Path,
    config_glob: str,
    output_root: Path,
    stop_on_error: bool,
) -> None:
    configs = _find_configs(config_glob)
    if len(configs) == 0:
        raise FileNotFoundError(f"No configs matched: {config_glob}")

    output_root.mkdir(parents=True, exist_ok=True)
    runner_script = Path(__file__).resolve().parent / "run_ho3d_all.py"
    if not runner_script.exists():
        raise FileNotFoundError(f"Runner script not found: {runner_script}")

    print(f"Found {len(configs)} ablation configs.")
    print(f"Data path: {data_path}")
    print(f"Output root: {output_root}")
    print(f"Runner script: {runner_script}")

    rows = []
    for idx, cfg in enumerate(configs, start=1):
        print(f"\n[{idx}/{len(configs)}] Running {cfg.name}")
        row = _run_single_config(
            python_exe=sys.executable,
            runner_script=runner_script,
            data_path=data_path,
            config_path=cfg,
            out_root=output_root,
        )
        rows.append(row)
        print(
            f"[{idx}/{len(configs)}] {cfg.name} -> {row['status']} "
            f"(rc={row['return_code']}, {row['elapsed_s']}s)"
        )
        print(f"  Log: {row['log_path']}")
        if row["summary_path"]:
            print(f"  Summary: {row['summary_path']}")

        if row["status"] != "ok" and stop_on_error:
            print("Stopping early due to --stop_on_error.")
            break

    _write_summary(output_root, rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run HO3D ablation configs (eccv_abla_*.yaml) and save each run into "
            "a matching output subfolder."
        )
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/justin/data/HO3D_V3/",
        help="Path to HO3D_V3 root directory.",
    )
    parser.add_argument(
        "--config_glob",
        type=str,
        default="configs/ho3d_exp/eccv_abla_*.yaml",
        help="Glob pattern for ablation config files.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/home/justin/results/eccv_point2pose/ablation",
        help="Base output directory for ablation runs.",
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop the sweep immediately if one config run fails.",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run_ho3d_ablation(
        data_path=Path(args.data_path).expanduser().resolve(),
        config_glob=args.config_glob,
        output_root=Path(args.output_root).expanduser().resolve(),
        stop_on_error=bool(args.stop_on_error),
    )
