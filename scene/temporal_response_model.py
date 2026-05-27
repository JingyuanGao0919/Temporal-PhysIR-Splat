import math
import os

import torch
import torch.nn as nn

from utils.system_utils import searchForMaxIteration


class TemporalResponseNetwork(nn.Module):
    def __init__(self, basis_count=4, gain_limit=0.80, bias_limit=0.15):
        super().__init__()
        self.basis_count = int(max(1, basis_count))
        self.gain_limit = float(gain_limit)
        self.bias_limit = float(bias_limit)
        self.gain_coeffs = nn.Parameter(torch.zeros(3, self.basis_count))
        self.bias_coeffs = nn.Parameter(torch.zeros(3, self.basis_count))
        self.thermal_gain = nn.Parameter(torch.zeros(3, 1))
        self.thermal_bias = nn.Parameter(torch.zeros(3, 1))

    def _basis(self, time_value, device, dtype):
        t = torch.as_tensor(time_value, device=device, dtype=dtype).reshape(1)
        centered = t - 0.5
        terms = [
            centered,
            centered * centered - (1.0 / 12.0),
            torch.sin(2.0 * math.pi * t),
            torch.cos(2.0 * math.pi * t),
        ]
        freq = 2
        while len(terms) < self.basis_count:
            terms.append(torch.sin(2.0 * math.pi * freq * t))
            if len(terms) >= self.basis_count:
                break
            terms.append(torch.cos(2.0 * math.pi * freq * t))
            freq += 1
        return torch.cat(terms[:self.basis_count], dim=0)

    def forward(self, image, time_value, thermal_delta=None):
        basis = self._basis(time_value, image.device, image.dtype)
        raw_gain = self.gain_coeffs @ basis
        raw_bias = self.bias_coeffs @ basis
        if thermal_delta is not None:
            delta = torch.as_tensor(thermal_delta, device=image.device, dtype=image.dtype).reshape(1)
            raw_gain = raw_gain + self.thermal_gain[:, 0] * delta
            raw_bias = raw_bias + self.thermal_bias[:, 0] * delta
        gain = 1.0 + self.gain_limit * torch.tanh(raw_gain)
        bias = self.bias_limit * torch.tanh(raw_bias)
        return image * gain.view(3, 1, 1) + bias.view(3, 1, 1)

    def regularization(self):
        return (
            self.gain_coeffs.square().mean()
            + self.bias_coeffs.square().mean()
            + self.thermal_gain.square().mean()
            + self.thermal_bias.square().mean()
        )


class TemporalResponseModel:
    def __init__(self, enabled=True, basis_count=4, gain_limit=0.80, bias_limit=0.15):
        self.enabled = bool(enabled)
        self.affine = TemporalResponseNetwork(basis_count, gain_limit, bias_limit).cuda()
        self.optimizer = None

    def step(self, image, time_value, gaussians=None):
        if not self.enabled:
            return image
        thermal_delta = None
        if gaussians is not None and getattr(gaussians, "has_thermal_temporal", False):
            luminance = gaussians.thermal_luminance(time_value).mean()
            ref_time = torch.tensor([0.5], device=luminance.device, dtype=luminance.dtype)
            ref_luminance = gaussians.thermal_luminance(ref_time).mean()
            thermal_delta = luminance - ref_luminance
        return self.affine(image, time_value, thermal_delta)

    def regularization(self, training_args):
        if not self.enabled:
            return torch.tensor(0.0, device="cuda")
        weight = float(getattr(training_args, "lambda_temporal_response", 0.0))
        if weight <= 0.0:
            return torch.tensor(0.0, device="cuda")
        return weight * self.affine.regularization()

    def train_setting(self, training_args):
        lr = float(getattr(training_args, "temporal_response_lr", 0.0))
        self.optimizer = torch.optim.Adam(
            [{"params": list(self.affine.parameters()), "lr": lr, "name": "temporal_response"}],
            lr=0.0,
            eps=1e-15,
        )

    def save_weights(self, model_path, iteration):
        if not self.enabled:
            return
        out_weights_path = os.path.join(model_path, "TPRC/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(
            {
                "enabled": self.enabled,
                "state_dict": self.affine.state_dict(),
                "basis_count": self.affine.basis_count,
                "gain_limit": self.affine.gain_limit,
                "bias_limit": self.affine.bias_limit,
            },
            os.path.join(out_weights_path, "tprc.pth"),
        )

    def _resolve_weights_path(self, model_path, iteration=-1):
        root = os.path.join(model_path, "TPRC")
        if not os.path.exists(root):
            return None
        loaded_iter = searchForMaxIteration(root) if iteration == -1 else iteration
        weights_path = os.path.join(root, "iteration_{}".format(loaded_iter), "tprc.pth")
        if os.path.exists(weights_path):
            return weights_path
        return None

    def load_weights(self, model_path, iteration=-1):
        weights_path = self._resolve_weights_path(model_path, iteration)
        if weights_path is None:
            return
        payload = torch.load(weights_path)
        if isinstance(payload, dict) and "state_dict" in payload:
            self.enabled = bool(payload.get("enabled", self.enabled))
            self.affine.load_state_dict(payload["state_dict"], strict=False)
        else:
            self.affine.load_state_dict(payload, strict=False)

    def update_learning_rate(self, iteration):
        return None
