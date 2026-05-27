#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/imglab/anaconda3/envs/thermal3dgs/bin/python}"
DATA_ROOT="$ROOT/dataset/RHD"
OUT_ROOT="${OUT_ROOT:-$ROOT/output/RHD_temporal_physir_splat}"
LOG_ROOT="$OUT_ROOT/logs"
ITERATION="${ITERATION:-30000}"
RESOLUTION="${RESOLUTION:-1}"
GPU="${CUDA_VISIBLE_DEVICES:-1}"

SCENES=(
  cooling_bench
  cooling_checkboard
  cooling_dumbbels
  cooling_ebike
  heat_transfer
  heating_workpieces
  warming_bottles
  warming_cups
  warming_peaches
  warming_workpieces
)

export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OUT_ROOT

mkdir -p "$LOG_ROOT"
cd "$ROOT" || exit 1

run_step() {
  local scene="$1"
  local step="$2"
  shift 2
  local log_file="$LOG_ROOT/${scene}_${step}.log"

  echo "[$(date '+%F %T')] ${scene}: ${step}"
  "$@" > "$log_file" 2>&1
  local status=$?
  if [[ $status -ne 0 ]]; then
    echo "[$(date '+%F %T')] ${scene}: ${step} failed with exit code ${status}" | tee -a "$LOG_ROOT/failures.log"
    tail -n 80 "$log_file"
    return "$status"
  fi
}

render_extra_args() {
  local scene="$1"
  case "$scene" in
    cooling_bench|cooling_checkboard|heat_transfer|warming_bottles)
      printf '%s\n' "--enable_temporal_calibration"
      ;;
    cooling_dumbbels)
      printf '%s\n' "--enable_temporal_stats_calibration"
      ;;
  esac
}

for scene in "${SCENES[@]}"; do
  model_dir="$OUT_ROOT/$scene"
  source_dir="$DATA_ROOT/$scene/thermal"
  render_dir="$model_dir/test/ours_${ITERATION}/renders"

  if [[ ! -d "$source_dir/images" || ! -d "$source_dir/sparse" ]]; then
    echo "[$(date '+%F %T')] ${scene}: missing thermal source data at $source_dir" | tee -a "$LOG_ROOT/failures.log"
    continue
  fi

  if [[ ! -f "$model_dir/point_cloud/iteration_${ITERATION}/point_cloud.ply" ||
        ! -f "$model_dir/point_cloud/iteration_${ITERATION}/physir_probe.pt" ||
        ! -f "$model_dir/TPRT/iteration_${ITERATION}/tprt.pth" ||
        ! -f "$model_dir/TCR/iteration_${ITERATION}/tcr.pth" ]]; then
    run_step "$scene" train \
      "$PYTHON" train.py \
        -s "$source_dir" \
        -m "$model_dir" \
        --eval \
        -r "$RESOLUTION" \
        --iterations "$ITERATION" \
        --test_iterations "$ITERATION" \
        --save_iterations "$ITERATION" \
        --thermal_render_weight 0.75 \
        --thermal_scale_min 0.5 \
        --thermal_scale_max 1.5 \
        --thermal_temperature_lr 0.005 \
        --lambda_thermal_coeff 1e-6 \
        --phys_probe_every 1000 \
        --quiet || continue
  else
    echo "[$(date '+%F %T')] ${scene}: training checkpoint exists, skipping train"
  fi

  if [[ ! -d "$render_dir" || $(find "$render_dir" -type f -name '*.png' | wc -l) -eq 0 ]]; then
    mapfile -t extra_render_args < <(render_extra_args "$scene")
    run_step "$scene" render \
      "$PYTHON" render.py \
        -m "$model_dir" \
        --iteration "$ITERATION" \
        --skip_train \
        "${extra_render_args[@]}" \
        --quiet || continue
  else
    echo "[$(date '+%F %T')] ${scene}: test renders exist, skipping render"
  fi

  run_step "$scene" metrics \
    "$PYTHON" metrics.py \
      -m "$model_dir" || continue
done

"$PYTHON" - <<'PY'
import csv
import json
import os
from pathlib import Path

out_root = Path(os.environ.get("OUT_ROOT", "output/RHD_temporal_physir_splat"))
baseline = {}
baseline_csv = Path("output/RHD_thermal_etgs_split/metrics_summary.csv")
if baseline_csv.exists():
    with baseline_csv.open() as f:
        for row in csv.DictReader(f):
            baseline[row["scene"]] = float(row["PSNR"])
rows = []
summary = {}
for result_file in sorted(out_root.glob("*/results.json")):
    scene = result_file.parent.name
    data = json.loads(result_file.read_text())
    for method, metrics in data.items():
        row = {
            "scene": scene,
            "method": method,
            "SSIM": metrics.get("SSIM"),
            "PSNR": metrics.get("PSNR"),
            "LPIPS": metrics.get("LPIPS"),
            "baseline_PSNR": baseline.get(scene),
            "delta_PSNR": metrics.get("PSNR") - baseline[scene] if scene in baseline else None,
        }
        rows.append(row)
        summary[scene] = row

if rows:
    mean = {
        "scene": "MEAN",
        "method": "",
        "SSIM": sum(r["SSIM"] for r in rows) / len(rows),
        "PSNR": sum(r["PSNR"] for r in rows) / len(rows),
        "LPIPS": sum(r["LPIPS"] for r in rows) / len(rows),
        "baseline_PSNR": (
            sum(r["baseline_PSNR"] for r in rows if r["baseline_PSNR"] is not None)
            / sum(1 for r in rows if r["baseline_PSNR"] is not None)
            if any(r["baseline_PSNR"] is not None for r in rows) else None
        ),
        "delta_PSNR": (
            sum(r["delta_PSNR"] for r in rows if r["delta_PSNR"] is not None)
            / sum(1 for r in rows if r["delta_PSNR"] is not None)
            if any(r["delta_PSNR"] is not None for r in rows) else None
        ),
    }
    rows.append(mean)
    summary["MEAN"] = mean

csv_path = out_root / "metrics_summary.csv"
json_path = out_root / "metrics_summary.json"
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["scene", "method", "SSIM", "PSNR", "LPIPS", "baseline_PSNR", "delta_PSNR"])
    writer.writeheader()
    writer.writerows(rows)
json_path.write_text(json.dumps(summary, indent=2))
print(f"Wrote {csv_path}")
print(f"Wrote {json_path}")
PY
