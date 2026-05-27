#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
import math
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation


class GaussianModel:
    def __init__(self, sh_degree: int):

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._thermal_temperature_base = torch.empty(0)
        self._thermal_temperature_coeffs = torch.empty(0)
        self._thermal_emissivity_raw = torch.empty(0)
        self._phys_temperature = torch.empty(0)
        self._phys_emissivity = torch.empty(0)
        self._phys_luminance = torch.empty(0)
        self._phys_iteration = -1
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)

        self.optimizer = None

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.thermal_temporal_enabled = False
        self.thermal_basis_count = 0
        self.thermal_min = 0.0
        self.thermal_max = 1.0
        self.thermal_env = 0.5
        self.thermal_render_weight = 0.0
        self.thermal_scale_min = 0.70
        self.thermal_scale_max = 1.30
        self.thermal_emissivity_init = 0.95
        self.thermal_emissivity_min = 0.70
        self.thermal_emissivity_max = 0.99

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def has_thermal_temporal(self):
        return (
            self.thermal_temporal_enabled
            and torch.is_tensor(self._thermal_temperature_base)
            and self._thermal_temperature_base.numel() > 0
            and torch.is_tensor(self._thermal_temperature_coeffs)
            and self._thermal_temperature_coeffs.numel() > 0
            and torch.is_tensor(self._thermal_emissivity_raw)
            and self._thermal_emissivity_raw.numel() > 0
        )

    @property
    def has_temporal_physir(self):
        return self.has_thermal_temporal

    @property
    def thermal_emissivity(self):
        if not torch.is_tensor(self._thermal_emissivity_raw) or self._thermal_emissivity_raw.numel() == 0:
            return torch.empty(0, device=self.get_xyz.device)
        span = max(self.thermal_emissivity_max - self.thermal_emissivity_min, 1e-6)
        return self.thermal_emissivity_min + span * torch.sigmoid(self._thermal_emissivity_raw)

    @property
    def temporal_physir_emissivity(self):
        return self.thermal_emissivity

    @property
    def get_physical_temperature(self):
        return self._phys_temperature

    @property
    def get_physical_emissivity(self):
        return self._phys_emissivity

    @property
    def get_physical_luminance(self):
        return self._phys_luminance

    def configure_temporal_physir(
        self,
        source_path,
        enable=True,
        basis_count=4,
        render_weight=0.35,
        scale_min=0.70,
        scale_max=1.30,
        emissivity_init=0.95,
        emissivity_min=0.70,
        emissivity_max=0.99,
    ):
        self.thermal_temporal_enabled = bool(enable)
        self.thermal_basis_count = max(1, int(basis_count))
        self.thermal_render_weight = float(render_weight)
        self.thermal_scale_min = float(scale_min)
        self.thermal_scale_max = float(scale_max)
        self.thermal_emissivity_min = float(emissivity_min)
        self.thermal_emissivity_max = float(emissivity_max)
        self.thermal_emissivity_init = float(
            min(max(emissivity_init, self.thermal_emissivity_min + 1e-4), self.thermal_emissivity_max - 1e-4)
        )

        info_path = os.path.join(str(source_path), "info.json")
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            self.thermal_min = float(info.get("min_value", 0.0))
            self.thermal_max = float(info.get("max_value", 1.0))
            self.thermal_env = float(info.get("T_env", 0.5 * (self.thermal_min + self.thermal_max)))
        except Exception as exc:
            print(f"[thermal-temporal] Could not read {info_path}: {exc}. Using unit temperature range.")
            self.thermal_min = 0.0
            self.thermal_max = 1.0
            self.thermal_env = 0.5

        if not (self.thermal_max > self.thermal_min):
            self.thermal_max = self.thermal_min + 1.0

    def _initial_temperature_from_colors(self, colors):
        if not self.thermal_temporal_enabled:
            return torch.empty((0, 1), device="cuda")
        weights = colors.new_tensor([0.299, 0.587, 0.114])
        gray = (colors[:, :3].clamp(0.0, 1.0) * weights).sum(dim=-1, keepdim=True)
        return self.thermal_min + gray * max(self.thermal_max - self.thermal_min, 1e-6)

    def _initial_emissivity_raw(self, count, device):
        e_min = self.thermal_emissivity_min
        e_max = self.thermal_emissivity_max
        frac = (self.thermal_emissivity_init - e_min) / max(e_max - e_min, 1e-6)
        frac = min(max(frac, 1e-4), 1.0 - 1e-4)
        raw = math.log(frac / (1.0 - frac))
        return torch.full((count, 1), raw, dtype=torch.float32, device=device)

    def _thermal_basis(self, time_value, device, dtype):
        if torch.is_tensor(time_value):
            t = time_value.to(device=device, dtype=dtype).reshape(1)
        else:
            t = torch.tensor([float(time_value)], device=device, dtype=dtype)
        t = torch.clamp(t, 0.0, 1.0)
        centered = t - 0.5
        basis = [
            centered,
            centered * centered - (1.0 / 12.0),
            torch.sin(2.0 * math.pi * t),
            torch.cos(2.0 * math.pi * t) - 1.0,
            torch.sin(4.0 * math.pi * t),
            torch.cos(4.0 * math.pi * t) - 1.0,
        ]
        if self.thermal_basis_count > len(basis):
            for k in range(len(basis), self.thermal_basis_count):
                basis.append(centered ** (k + 1))
        return torch.stack(basis[: self.thermal_basis_count], dim=0).reshape(-1, 1)

    def thermal_temperature(self, time_value):
        if not self.has_thermal_temporal:
            return torch.empty(0, device=self.get_xyz.device)
        basis = self._thermal_basis(time_value, self._thermal_temperature_base.device, self._thermal_temperature_base.dtype)
        delta = self._thermal_temperature_coeffs @ basis
        return self._thermal_temperature_base + delta

    def temporal_physir_temperature(self, time_value):
        return self.thermal_temperature(time_value)

    def thermal_luminance(self, time_value):
        if not self.has_thermal_temporal:
            return torch.empty(0, device=self.get_xyz.device)
        temperature = self.thermal_temperature(time_value)
        span = max(self.thermal_max - self.thermal_min, 1e-6)
        thermal_gray = ((temperature - self.thermal_min) / span).clamp(0.0, 1.0)
        ambient_gray = min(max((self.thermal_env - self.thermal_min) / span, 0.0), 1.0)
        emissivity = self.thermal_emissivity
        return emissivity * thermal_gray + (1.0 - emissivity) * ambient_gray

    def temporal_physir_luminance(self, time_value):
        return self.thermal_luminance(time_value)

    def thermal_scale(self, time_value, render_weight=None, scale_min=None, scale_max=None):
        if not self.has_thermal_temporal:
            return 1.0
        weight = self.thermal_render_weight if render_weight is None else float(render_weight)
        if weight <= 0.0:
            return 1.0
        q_t = self.thermal_luminance(time_value)
        q_ref = self.thermal_luminance(torch.tensor([0.5], device=q_t.device, dtype=q_t.dtype)).clamp_min(1e-4)
        raw_scale = q_t / q_ref
        lo = self.thermal_scale_min if scale_min is None else float(scale_min)
        hi = self.thermal_scale_max if scale_max is None else float(scale_max)
        return (1.0 + weight * (raw_scale - 1.0)).clamp(lo, hi)

    def temporal_physir_scale(self, time_value, render_weight=None, scale_min=None, scale_max=None):
        return self.thermal_scale(time_value, render_weight, scale_min, scale_max)

    def thermal_regularization(self, training_args):
        if not self.has_thermal_temporal:
            return self.get_xyz.new_tensor(0.0)
        reg = self.get_xyz.new_tensor(0.0)
        lambda_coeff = float(getattr(training_args, "lambda_thermal_coeff", 0.0))
        if lambda_coeff > 0:
            reg = reg + lambda_coeff * self._thermal_temperature_coeffs.square().mean()
        lambda_bounds = float(getattr(training_args, "lambda_thermal_temp_bounds", 0.0))
        if lambda_bounds > 0:
            margin = 0.05 * max(self.thermal_max - self.thermal_min, 1e-6)
            low = self.thermal_min - margin
            high = self.thermal_max + margin
            temp = self._thermal_temperature_base
            reg = reg + lambda_bounds * (torch.relu(low - temp).square().mean() + torch.relu(temp - high).square().mean())
        lambda_emiss = float(getattr(training_args, "lambda_emissivity_prior", 0.0))
        if lambda_emiss > 0:
            target = self.thermal_emissivity.new_full(self.thermal_emissivity.shape, self.thermal_emissivity_init)
            reg = reg + lambda_emiss * (self.thermal_emissivity - target).square().mean()
        return reg

    def temporal_physir_regularization(self, training_args):
        return self.thermal_regularization(training_args)

    @torch.no_grad()
    def fit_physir_probe(self, iteration=-1, time_value=0.5):
        if not self.has_thermal_temporal:
            return
        self._phys_iteration = int(iteration)
        self._phys_temperature = self.thermal_temperature(time_value).detach().clone()
        self._phys_emissivity = self.thermal_emissivity.detach().clone()
        self._phys_luminance = self.thermal_luminance(time_value).detach().clone()

    def save_physir_probe(self, path):
        if not torch.is_tensor(self._phys_temperature) or self._phys_temperature.numel() == 0:
            return
        torch.save(
            {
                "iteration": self._phys_iteration,
                "temperature": self._phys_temperature.detach().cpu(),
                "emissivity": self._phys_emissivity.detach().cpu(),
                "luminance": self._phys_luminance.detach().cpu(),
                "temperature_min": self.thermal_min,
                "temperature_max": self.thermal_max,
                "environment_temperature": self.thermal_env,
            },
            path,
        )

    def fit_temporal_physir_probe(self, iteration=-1, time_value=0.5):
        return self.fit_physir_probe(iteration, time_value)

    def save_temporal_physir_probe(self, path):
        return self.save_physir_probe(path)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = 5
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        pcd_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
        fused_color = RGB2SH(pcd_color)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        if self.thermal_temporal_enabled:
            temp_base = self._initial_temperature_from_colors(pcd_color)
            temp_coeffs = torch.zeros((fused_point_cloud.shape[0], self.thermal_basis_count), dtype=torch.float32, device="cuda")
            emissivity_raw = self._initial_emissivity_raw(fused_point_cloud.shape[0], "cuda")
            self._thermal_temperature_base = nn.Parameter(temp_base.requires_grad_(True))
            self._thermal_temperature_coeffs = nn.Parameter(temp_coeffs.requires_grad_(True))
            self._thermal_emissivity_raw = nn.Parameter(emissivity_raw.requires_grad_(True))
        else:
            self._thermal_temperature_base = torch.empty(0, device="cuda")
            self._thermal_temperature_coeffs = torch.empty(0, device="cuda")
            self._thermal_emissivity_raw = torch.empty(0, device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.spatial_lr_scale = 5

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr * self.spatial_lr_scale, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        if self.has_thermal_temporal:
            l.extend([
                {'params': [self._thermal_temperature_base], 'lr': training_args.thermal_temperature_lr, "name": "thermal_temp_base"},
                {'params': [self._thermal_temperature_coeffs], 'lr': training_args.thermal_temperature_lr, "name": "thermal_temp_coeffs"},
                {'params': [self._thermal_emissivity_raw], 'lr': training_args.thermal_emissivity_lr, "name": "thermal_emissivity"},
            ])

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        if self.has_thermal_temporal:
            l.append('thermal_temp_base')
            for i in range(self._thermal_temperature_coeffs.shape[1]):
                l.append('thermal_temp_coeff_{}'.format(i))
            l.append('thermal_emissivity_raw')
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = [xyz, normals, f_dc, f_rest, opacities, scale, rotation]
        if self.has_thermal_temporal:
            temp_base = self._thermal_temperature_base.detach().cpu().numpy()
            temp_coeffs = self._thermal_temperature_coeffs.detach().cpu().numpy()
            emissivity_raw = self._thermal_emissivity_raw.detach().cpu().numpy()
            attributes.extend([temp_base, temp_coeffs, emissivity_raw])
        attributes = np.concatenate(attributes, axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, og_number_points=-1):
        self.og_number_points = og_number_points
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        names = plydata.elements[0].data.dtype.names
        has_saved_thermal = (
            "thermal_temp_base" in names
            and "thermal_emissivity_raw" in names
            and any(name.startswith("thermal_temp_coeff_") for name in names)
        )
        if has_saved_thermal:
            coeff_names = [name for name in names if name.startswith("thermal_temp_coeff_")]
            coeff_names = sorted(coeff_names, key=lambda x: int(x.split("_")[-1]))
            temp_base = np.asarray(plydata.elements[0]["thermal_temp_base"])[..., np.newaxis].astype(np.float32)
            coeffs = np.zeros((xyz.shape[0], len(coeff_names)), dtype=np.float32)
            for idx, attr_name in enumerate(coeff_names):
                coeffs[:, idx] = np.asarray(plydata.elements[0][attr_name])
            emissivity_raw = np.asarray(plydata.elements[0]["thermal_emissivity_raw"])[..., np.newaxis].astype(np.float32)
            self.thermal_temporal_enabled = True
            self.thermal_basis_count = len(coeff_names)
            self._thermal_temperature_base = nn.Parameter(torch.tensor(temp_base, dtype=torch.float32, device="cuda").requires_grad_(True))
            self._thermal_temperature_coeffs = nn.Parameter(torch.tensor(coeffs, dtype=torch.float32, device="cuda").requires_grad_(True))
            self._thermal_emissivity_raw = nn.Parameter(torch.tensor(emissivity_raw, dtype=torch.float32, device="cuda").requires_grad_(True))
        else:
            self._thermal_temperature_base = torch.empty(0, device="cuda")
            self._thermal_temperature_coeffs = torch.empty(0, device="cuda")
            self._thermal_emissivity_raw = torch.empty(0, device="cuda")

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.has_thermal_temporal:
            self._thermal_temperature_base = optimizable_tensors["thermal_temp_base"]
            self._thermal_temperature_coeffs = optimizable_tensors["thermal_temp_coeffs"]
            self._thermal_emissivity_raw = optimizable_tensors["thermal_emissivity"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation, new_temp_base=None, new_temp_coeffs=None, new_emissivity_raw=None):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation}
        if self.has_thermal_temporal:
            d["thermal_temp_base"] = new_temp_base
            d["thermal_temp_coeffs"] = new_temp_coeffs
            d["thermal_emissivity"] = new_emissivity_raw

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.has_thermal_temporal:
            self._thermal_temperature_base = optimizable_tensors["thermal_temp_base"]
            self._thermal_temperature_coeffs = optimizable_tensors["thermal_temp_coeffs"]
            self._thermal_emissivity_raw = optimizable_tensors["thermal_emissivity"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        if self.has_thermal_temporal:
            new_temp_base = self._thermal_temperature_base[selected_pts_mask].repeat(N, 1)
            new_temp_coeffs = self._thermal_temperature_coeffs[selected_pts_mask].repeat(N, 1)
            new_emissivity_raw = self._thermal_emissivity_raw[selected_pts_mask].repeat(N, 1)
        else:
            new_temp_base = new_temp_coeffs = new_emissivity_raw = None

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation,
                                   new_temp_base, new_temp_coeffs, new_emissivity_raw)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        if self.has_thermal_temporal:
            new_temp_base = self._thermal_temperature_base[selected_pts_mask]
            new_temp_coeffs = self._thermal_temperature_coeffs[selected_pts_mask]
            new_emissivity_raw = self._thermal_emissivity_raw[selected_pts_mask]
        else:
            new_temp_base = new_temp_coeffs = new_emissivity_raw = None

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation, new_temp_base, new_temp_coeffs, new_emissivity_raw)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        self.denom[update_filter] += 1
