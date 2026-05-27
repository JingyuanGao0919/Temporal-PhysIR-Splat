import json
import os
import sys

import torch


def _stats(tensor):
    tensor = tensor.detach().float().reshape(-1)
    if tensor.numel() == 0:
        return None
    return {
        "mean": float(tensor.mean().item()),
        "p5": float(torch.quantile(tensor, 0.05).item()),
        "p95": float(torch.quantile(tensor, 0.95).item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


@torch.no_grad()
def temporal_physir_summary(iteration, gaussians, time_value):
    if not getattr(gaussians, "has_temporal_physir", getattr(gaussians, "has_thermal_temporal", False)):
        return None

    if hasattr(gaussians, "temporal_physir_temperature"):
        temperature = gaussians.temporal_physir_temperature(time_value)
        emissivity = gaussians.temporal_physir_emissivity
        luminance = gaussians.temporal_physir_luminance(time_value)
    else:
        temperature = gaussians.thermal_temperature(time_value)
        emissivity = gaussians.thermal_emissivity
        luminance = gaussians.thermal_luminance(time_value)

    summary = {
        "iteration": int(iteration),
        "gaussians": int(temperature.numel()),
        "time": float(torch.as_tensor(time_value).detach().cpu().reshape(-1)[0].item()),
        "temperature": _stats(temperature),
        "emissivity": _stats(emissivity),
        "luminance": _stats(luminance),
    }
    return summary


def summary_to_text(summary):
    if summary is None:
        return None
    temperature = summary["temperature"]
    emissivity = summary["emissivity"]
    return (
        "[ITER {iteration}] PhysIRProbe | T mean {t_mean:.4f} p5 {t_p5:.4f} p95 {t_p95:.4f} "
        "| emissivity mean {e_mean:.4f} p5 {e_p5:.4f} p95 {e_p95:.4f}"
    ).format(
        iteration=summary["iteration"],
        t_mean=temperature["mean"],
        t_p5=temperature["p5"],
        t_p95=temperature["p95"],
        e_mean=emissivity["mean"],
        e_p5=emissivity["p5"],
        e_p95=emissivity["p95"],
    )


def write_summary(model_path, summary, text):
    log_dir = os.path.join(model_path, "physir_probe")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "probe_log.jsonl"), "a", encoding="utf-8") as jsonl:
        jsonl.write(json.dumps(summary, sort_keys=True) + "\n")
    with open(os.path.join(log_dir, "result.txt"), "a", encoding="utf-8") as txt:
        txt.write(text + "\n")


def log_temporal_physir_probe(iteration, gaussians, time_value, model_path):
    summary = temporal_physir_summary(iteration, gaussians, time_value)
    text = summary_to_text(summary)
    if text is None:
        return
    sys.__stdout__.write("\n" + text + "\n")
    sys.__stdout__.flush()
    write_summary(model_path, summary, text)
