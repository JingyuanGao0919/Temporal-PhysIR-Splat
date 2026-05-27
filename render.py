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
from scene import Scene, TPRTModel, TCRModel, TemporalResponseModel
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.pose_utils import pose_spherical, render_wander_path
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import imageio
import numpy as np
import math
import time


def physir_transmittance(absorption, scattering, distance):
    exponent = torch.clamp((absorption + scattering) * distance, min=-10.0, max=10.0)
    return torch.exp(exponent)


def camera_time(view, use_timestamp_time=False):
    if use_timestamp_time and hasattr(view, "timestamp"):
        return view.timestamp
    return view.fid


def use_real_timestamp(dataset):
    return getattr(dataset, "use_timestamp_time", False) and not getattr(dataset, "disable_timestamp_time", False)


def render_camera_image(view, gaussians, pipeline, background, tprt, tcr, response_model, use_timestamp_time):
    time_value = camera_time(view, use_timestamp_time)
    xyz = gaussians.get_xyz
    time_input = time_value.unsqueeze(0).expand(xyz.shape[0], -1)
    absorption, scattering, distance = tprt.step(xyz, time_input)
    d_rgb = physir_transmittance(absorption, scattering, distance)
    d_rgb = d_rgb * gaussians.temporal_physir_scale(time_value)
    results = render(view, gaussians, pipeline, background, d_rgb)
    image = results["render"] + tcr.step(results["render"])
    if response_model is not None:
        image = response_model.step(image, time_value, gaussians)
        if response_model.enabled:
            image = torch.clamp(image, 0.0, 1.0)
    return image, time_value


def fit_channel_affine(rendered, gt):
    x = rendered.detach().clamp(0.0, 1.0).permute(1, 2, 0).reshape(-1, 3).cpu().numpy()
    y = gt.detach().clamp(0.0, 1.0).permute(1, 2, 0).reshape(-1, 3).cpu().numpy()
    gain = np.ones(3, dtype=np.float32)
    bias = np.zeros(3, dtype=np.float32)
    ones = np.ones((x.shape[0], 1), dtype=np.float32)
    for channel in range(3):
        design = np.concatenate([x[:, channel:channel + 1], ones], axis=1)
        coeff, _, _, _ = np.linalg.lstsq(design, y[:, channel], rcond=None)
        gain[channel] = np.clip(coeff[0], 0.2, 2.5)
        bias[channel] = np.clip(coeff[1], -0.5, 0.5)
    return gain, bias


def build_temporal_calibration(views, gaussians, pipeline, background, tprt, tcr, response_model, use_timestamp_time):
    if not views:
        return None
    samples = []
    for view in tqdm(views, desc="Calibrating temporal response"):
        image, time_value = render_camera_image(
            view, gaussians, pipeline, background, tprt, tcr, response_model, use_timestamp_time)
        gt = view.original_image[0:3, :, :].to(image.device)
        gain, bias = fit_channel_affine(image, gt)
        samples.append((float(time_value.detach().cpu().item()), gain, bias))
    samples.sort(key=lambda item: item[0])
    return {
        "time": np.asarray([item[0] for item in samples], dtype=np.float32),
        "gain": np.stack([item[1] for item in samples], axis=0),
        "bias": np.stack([item[2] for item in samples], axis=0),
    }


def build_temporal_stats_calibration(views, use_timestamp_time):
    if not views:
        return None
    samples = []
    for view in views:
        image = view.original_image[0:3, :, :].detach().clamp(0.0, 1.0)
        mean = image.mean(dim=(1, 2)).cpu().numpy().astype(np.float32)
        std = image.std(dim=(1, 2)).clamp_min(1e-6).cpu().numpy().astype(np.float32)
        time_value = camera_time(view, use_timestamp_time)
        samples.append((float(time_value.detach().cpu().item()), mean, std))
    samples.sort(key=lambda item: item[0])
    return {
        "time": np.asarray([item[0] for item in samples], dtype=np.float32),
        "mean": np.stack([item[1] for item in samples], axis=0),
        "std": np.stack([item[2] for item in samples], axis=0),
    }


def interpolate_temporal_stats(time_value, calibration):
    t = float(time_value.detach().cpu().item())
    times = calibration["time"]
    idx = int(np.searchsorted(times, t))
    if idx <= 0:
        return calibration["mean"][0], calibration["std"][0]
    if idx >= len(times):
        return calibration["mean"][-1], calibration["std"][-1]
    t0, t1 = times[idx - 1], times[idx]
    ratio = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
    mean = (1.0 - ratio) * calibration["mean"][idx - 1] + ratio * calibration["mean"][idx]
    std = (1.0 - ratio) * calibration["std"][idx - 1] + ratio * calibration["std"][idx]
    return mean, std


def apply_temporal_calibration(image, time_value, calibration):
    if calibration is None:
        return image
    t = float(time_value.detach().cpu().item())
    times = calibration["time"]
    idx = int(np.searchsorted(times, t))
    if idx <= 0:
        gain = calibration["gain"][0]
        bias = calibration["bias"][0]
    elif idx >= len(times):
        gain = calibration["gain"][-1]
        bias = calibration["bias"][-1]
    else:
        t0, t1 = times[idx - 1], times[idx]
        ratio = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        gain = (1.0 - ratio) * calibration["gain"][idx - 1] + ratio * calibration["gain"][idx]
        bias = (1.0 - ratio) * calibration["bias"][idx - 1] + ratio * calibration["bias"][idx]
    gain = torch.tensor(gain, dtype=image.dtype, device=image.device).view(3, 1, 1)
    bias = torch.tensor(bias, dtype=image.dtype, device=image.device).view(3, 1, 1)
    return torch.clamp(image * gain + bias, 0.0, 1.0)


def apply_temporal_stats_calibration(image, time_value, calibration):
    if calibration is None:
        return image
    target_mean, target_std = interpolate_temporal_stats(time_value, calibration)
    target_mean = torch.tensor(target_mean, dtype=image.dtype, device=image.device).view(3, 1, 1)
    target_std = torch.tensor(target_std, dtype=image.dtype, device=image.device).view(3, 1, 1)
    image_mean = image.mean(dim=(1, 2), keepdim=True)
    image_std = image.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    return torch.clamp((image - image_mean) * (target_std / image_std) + target_mean, 0.0, 1.0)


def render_set(model_path, load2gpt_on_the_fly, name, iteration, views, gaussians, pipeline, background, tprt, tcr,
               response_model=None, use_timestamp_time=False, temporal_calibration=None,
               temporal_stats_calibration=None):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    render_time_list = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if load2gpt_on_the_fly:
            view.load2gpu()
        torch.cuda.synchronize()
        start = time.time()
        rendering, time_value = render_camera_image(
            view, gaussians, pipeline, background, tprt, tcr, response_model, use_timestamp_time)
        rendering = apply_temporal_calibration(rendering, time_value, temporal_calibration)
        rendering = apply_temporal_stats_calibration(rendering, time_value, temporal_stats_calibration)
        torch.cuda.synchronize()
        end = time.time()
        render_time_list.append((end-start)*1000)
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        
    with open(os.path.join(model_path, name, "ours_{}".format(iteration), 'render_time.txt'), 'w') as f:
        for t in render_time_list:
            f.write("%.2fms\n"%t)
        f.write("Mean time: %.2fms\n"%(np.mean(render_time_list[5:])))


def interpolate_time(model_path, load2gpt_on_the_fly, name, iteration, views, gaussians, pipeline, background, tprt, tcr,
                     response_model=None, use_timestamp_time=False, temporal_calibration=None,
                     temporal_stats_calibration=None):
    render_path = os.path.join(model_path, name, "interpolate_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    frame = 1000
    idx = torch.randint(0, len(views), (1,)).item()
    view = views[idx]
    renderings = []
    for t in tqdm(range(0, frame, 1), desc="Rendering progress"):
        fid = torch.Tensor([t / (frame - 1)]).cuda()
        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        absorption, scattering, distance = tprt.step(xyz.detach(), time_input)
        d_rgb = physir_transmittance(absorption, scattering, distance)
        d_rgb = d_rgb * gaussians.temporal_physir_scale(fid)
        results = render(view, gaussians, pipeline, background, d_rgb)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(t) + ".png"))
    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)


def interpolate_all(model_path, load2gpt_on_the_fly, name, iteration, views, gaussians, pipeline, background, tprt, tcr,
                    response_model=None, use_timestamp_time=False, temporal_calibration=None,
                    temporal_stats_calibration=None):
    render_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "renders")
    makedirs(render_path, exist_ok=True)


    frame = 150
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, frame + 1)[:-1]],
                               0)
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = torch.randint(0, len(views), (1,)).item()
    view = views[idx]  # Choose a specific time for rendering

    renderings = []
    for i, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        fid = torch.Tensor([i / (frame - 1)]).cuda()

        matrix = np.linalg.inv(np.array(pose))
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        view.reset_extrinsic(R, T)

        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)

        absorption, scattering, distance = tprt.step(xyz.detach(), time_input)
        d_rgb = physir_transmittance(absorption, scattering, distance)
        d_rgb = d_rgb * gaussians.temporal_physir_scale(fid)
        results = render(view, gaussians, pipeline, background, d_rgb)
        results['render'] = results['render'] + tcr.step(results['render'])
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))


        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(i) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)

def interpolate_view_original(model_path, load2gpt_on_the_fly, name, iteration, views, gaussians, pipeline, background,
                              tprt, tcr, response_model=None, use_timestamp_time=False, temporal_calibration=None,
                              temporal_stats_calibration=None):
    render_path = os.path.join(model_path, name, "interpolate_hyper_view_{}".format(iteration), "renders")


    makedirs(render_path, exist_ok=True)


    frame = 1000
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    R = []
    T = []
    for view in views:
        R.append(view.R)
        T.append(view.T)

    view = views[0]
    renderings = []
    for i in tqdm(range(frame), desc="Rendering progress"):
        fid = torch.Tensor([i / (frame - 1)]).cuda()

        query_idx = i / frame * len(views)
        begin_idx = int(np.floor(query_idx))
        end_idx = int(np.ceil(query_idx))
        if end_idx == len(views):
            break
        view_begin = views[begin_idx]
        view_end = views[end_idx]
        R_begin = view_begin.R
        R_end = view_end.R
        t_begin = view_begin.T
        t_end = view_end.T

        ratio = query_idx - begin_idx

        R_cur = (1 - ratio) * R_begin + ratio * R_end
        T_cur = (1 - ratio) * t_begin + ratio * t_end

        view.reset_extrinsic(R_cur, T_cur)

        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        absorption, scattering, distance = tprt.step(xyz, time_input)
        torch.cuda.synchronize() 
        d_rgb = physir_transmittance(absorption, scattering, distance)
        d_rgb = d_rgb * gaussians.temporal_physir_scale(fid)

        results = render(view, gaussians, pipeline, background, d_rgb)
        #import pdb;pdb.set_trace()
        results['render'] = results['render'] + tcr.step(results['render'])
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=60, quality=8)





def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool,
                mode: str):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        gaussians.configure_temporal_physir(
            dataset.source_path,
            enable=getattr(dataset, "enable_thermal_temporal", False),
            basis_count=getattr(dataset, "thermal_basis_count", 4),
            render_weight=getattr(dataset, "thermal_render_weight", 0.0),
            scale_min=getattr(dataset, "thermal_scale_min", 0.70),
            scale_max=getattr(dataset, "thermal_scale_max", 1.30),
            emissivity_init=getattr(dataset, "thermal_emissivity_init", 0.95),
            emissivity_min=getattr(dataset, "thermal_emissivity_min", 0.70),
            emissivity_max=getattr(dataset, "thermal_emissivity_max", 0.99),
        )
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        tprt = TPRTModel(dataset.is_blender)
        tprt.load_weights(dataset.model_path)
        tcr = TCRModel()
        tcr.load_weights(dataset.model_path)
        response_model = TemporalResponseModel(
            enabled=getattr(dataset, "enable_temporal_response", False),
            basis_count=getattr(dataset, "thermal_basis_count", 4),
            gain_limit=getattr(dataset, "temporal_response_gain_limit", 0.80),
            bias_limit=getattr(dataset, "temporal_response_bias_limit", 0.15),
        )
        response_model.load_weights(dataset.model_path)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if mode == "render":
            render_func = render_set
        elif mode == "time":
            render_func = interpolate_time
        elif mode == "view":
            render_func = interpolate_view
        elif mode == "pose":
            render_func = interpolate_poses
        elif mode == "original":
            render_func = interpolate_view_original
        else:
            render_func = interpolate_all

        temporal_calibration = None
        if mode == "render" and not skip_test and getattr(dataset, "enable_temporal_calibration", False):
            temporal_calibration = build_temporal_calibration(
                scene.getTrainCameras(), gaussians, pipeline, background, tprt, tcr,
                response_model, use_real_timestamp(dataset))
        temporal_stats_calibration = None
        if mode == "render" and not skip_test and getattr(dataset, "enable_temporal_stats_calibration", False):
            temporal_stats_calibration = build_temporal_stats_calibration(
                scene.getTrainCameras(), use_real_timestamp(dataset))

        if not skip_train:
            render_func(dataset.model_path, dataset.load2gpu_on_the_fly, "train", scene.loaded_iter,
                        scene.getTrainCameras(), gaussians, pipeline,
                        background, tprt, tcr, response_model, use_real_timestamp(dataset))

        if not skip_test:
            render_func(dataset.model_path, dataset.load2gpu_on_the_fly, "test", scene.loaded_iter,
                        scene.getTestCameras(), gaussians, pipeline,
                        background, tprt, tcr, response_model, use_real_timestamp(dataset), temporal_calibration,
                        temporal_stats_calibration)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", default=True, action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mode", default='render', choices=['render', 'time', 'view', 'all', 'pose', 'original'])
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.mode)
