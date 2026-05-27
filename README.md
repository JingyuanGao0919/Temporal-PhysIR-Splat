# Temporal PhysIR-Splat

This repository contains the temporal branch of **PhysIR-Splat: Physically Consistent Thermal Infrared Radiative Transfer in 3D Gaussian Splatting**.

Temporal PhysIR-Splat targets dynamic thermal infrared novel-view synthesis. It extends PhysIR-Splat from static thermal scenes to time-varying thermal processes by assigning Gaussian primitives physically interpretable temperature and emissivity attributes, then modeling temperature evolution over continuous timestamps.

## Method Overview

Temporal PhysIR-Splat represents thermal infrared image formation with Gaussian primitives carrying temperature, effective in-band emissivity, and opacity. During rendering, the temporal thermal state modulates the infrared radiance of each Gaussian. The temporal branch uses learnable basis coefficients to evaluate Gaussian temperature at arbitrary timestamps without numerical ODE integration.

## RHD Results

The gallery shows representative RHD test renderings. Each sample shows grayscale / RHD-style pseudo-color. `t` is the normalized timestamp used by the temporal renderer.

<table>
<tr>
<td width="50%"><img src="assets/rhd_results/cooling_bench_top1_00012.png" width="176"> <img src="assets/rhd_results_pseudo/cooling_bench_top1_00012.png" width="176"><br><sub>gray / RHD pseudo</sub><br><b>cooling_bench</b> / view 00012.png<br>t=0.6043, PSNR=46.43, SSIM=0.9939, LPIPS=0.0340</td>
<td width="50%"><img src="assets/rhd_results/cooling_bench_top2_00023.png" width="176"> <img src="assets/rhd_results_pseudo/cooling_bench_top2_00023.png" width="176"><br><sub>gray / RHD pseudo</sub><br><b>cooling_bench</b> / view 00023.png<br>t=0.8952, PSNR=46.22, SSIM=0.9948, LPIPS=0.0288</td>
</tr>
<tr>
<td width="50%"><img src="assets/rhd_results/cooling_checkboard_top1_00017.png" width="176"> <img src="assets/rhd_results_pseudo/cooling_checkboard_top1_00017.png" width="176"><br><sub>gray / RHD pseudo</sub><br><b>cooling_checkboard</b> / view 00017.png<br>t=0.6460, PSNR=50.76, SSIM=0.9950, LPIPS=0.0258</td>
<td width="50%"><img src="assets/rhd_results/cooling_checkboard_top2_00016.png" width="176"> <img src="assets/rhd_results_pseudo/cooling_checkboard_top2_00016.png" width="176"><br><sub>gray / RHD pseudo</sub><br><b>cooling_checkboard</b> / view 00016.png<br>t=0.6436, PSNR=50.68, SSIM=0.9951, LPIPS=0.0299</td>
</tr>
<tr>
<td width="50%"><img src="assets/rhd_results/cooling_dumbbels_top1_00004.png" width="176"> <img src="assets/rhd_results_pseudo/cooling_dumbbels_top1_00004.png" width="176"><br><sub>gray / RHD pseudo</sub><br><b>cooling_dumbbels</b> / view 00004.png<br>t=0.0878, PSNR=44.13, SSIM=0.9944, LPIPS=0.0193</td>
<td width="50%"><img src="assets/rhd_results/cooling_dumbbels_top2_00023.png" width="176"> <img src="assets/rhd_results_pseudo/cooling_dumbbels_top2_00023.png" width="176"><br><sub>gray / RHD pseudo</sub><br><b>cooling_dumbbels</b> / view 00023.png<br>t=0.8137, PSNR=43.55, SSIM=0.9951, LPIPS=0.0179</td>
</tr>
<tr>
<td width="50%"><img src="assets/rhd_results/cooling_ebike_top1_00027.png" width="176"> <img src="assets/rhd_results_pseudo/cooling_ebike_top1_00027.png" width="176"><br><sub>gray / RHD pseudo</sub><br><b>cooling_ebike</b> / view 00027.png<br>t=0.8617, PSNR=48.55, SSIM=0.9957, LPIPS=0.0107</td>
<td width="50%"><img src="assets/rhd_results/cooling_ebike_top2_00036.png" width="176"> <img src="assets/rhd_results_pseudo/cooling_ebike_top2_00036.png" width="176"><br><sub>gray / RHD pseudo</sub><br><b>cooling_ebike</b> / view 00036.png<br>t=0.8861, PSNR=47.54, SSIM=0.9953, LPIPS=0.0126</td>
</tr>
<tr>
<td width="50%"><img src="assets/rhd_results/heat_transfer_top1_00010.png" width="176"> <img src="assets/rhd_results_pseudo/heat_transfer_top1_00010.png" width="176"><br><sub>gray / RHD pseudo</sub><br><b>heat_transfer</b> / view 00010.png<br>t=0.3510, PSNR=46.87, SSIM=0.9949, LPIPS=0.0403</td>
<td width="50%"><img src="assets/rhd_results/heat_transfer_top2_00009.png" width="176"> <img src="assets/rhd_results_pseudo/heat_transfer_top2_00009.png" width="176"><br><sub>gray / RHD pseudo</sub><br><b>heat_transfer</b> / view 00009.png<br>t=0.3485, PSNR=46.74, SSIM=0.9957, LPIPS=0.0376</td>
</tr>
<tr>
<td width="50%"><img src="assets/rhd_results/heating_workpieces_top1_00013.png" width="176"> <img src="assets/rhd_results_pseudo/heating_workpieces_top1_00013.png" width="176"><br><sub>gray / RHD pseudo</sub><br><b>heating_workpieces</b> / view 00013.png<br>t=0.4438, PSNR=45.82, SSIM=0.9960, LPIPS=0.0177</td>
<td width="50%"><img src="assets/rhd_results/heating_workpieces_top2_00007.png" width="176"> <img src="assets/rhd_results_pseudo/heating_workpieces_top2_00007.png" width="176"><br><sub>gray / RHD pseudo</sub><br><b>heating_workpieces</b> / view 00007.png<br>t=0.4243, PSNR=44.77, SSIM=0.9961, LPIPS=0.0165</td>
</tr>
<tr>
<td width="50%"><img src="assets/rhd_results/warming_bottles_top1_00016.png" width="176"> <img src="assets/rhd_results_pseudo/warming_bottles_top1_00016.png" width="176"><br><sub>gray / RHD pseudo</sub><br><b>warming_bottles</b> / view 00016.png<br>t=0.7008, PSNR=44.98, SSIM=0.9938, LPIPS=0.0837</td>
<td width="50%"><img src="assets/rhd_results/warming_bottles_top2_00014.png" width="176"> <img src="assets/rhd_results_pseudo/warming_bottles_top2_00014.png" width="176"><br><sub>gray / RHD pseudo</sub><br><b>warming_bottles</b> / view 00014.png<br>t=0.6920, PSNR=44.94, SSIM=0.9938, LPIPS=0.0879</td>
</tr>
<tr>
<td width="50%"><img src="assets/rhd_results/warming_cups_top1_00013.png" width="176"> <img src="assets/rhd_results_pseudo/warming_cups_top1_00013.png" width="176"><br><sub>gray / RHD pseudo</sub><br><b>warming_cups</b> / view 00013.png<br>t=0.3484, PSNR=46.94, SSIM=0.9951, LPIPS=0.0656</td>
<td width="50%"><img src="assets/rhd_results/warming_cups_top2_00012.png" width="176"> <img src="assets/rhd_results_pseudo/warming_cups_top2_00012.png" width="176"><br><sub>gray / RHD pseudo</sub><br><b>warming_cups</b> / view 00012.png<br>t=0.3453, PSNR=46.94, SSIM=0.9949, LPIPS=0.0695</td>
</tr>
<tr>
<td width="50%"><img src="assets/rhd_results/warming_peaches_top1_00017.png" width="176"> <img src="assets/rhd_results_pseudo/warming_peaches_top1_00017.png" width="176"><br><sub>gray / RHD pseudo</sub><br><b>warming_peaches</b> / view 00017.png<br>t=0.6210, PSNR=46.72, SSIM=0.9954, LPIPS=0.0925</td>
<td width="50%"><img src="assets/rhd_results/warming_peaches_top2_00020.png" width="176"> <img src="assets/rhd_results_pseudo/warming_peaches_top2_00020.png" width="176"><br><sub>gray / RHD pseudo</sub><br><b>warming_peaches</b> / view 00020.png<br>t=0.6283, PSNR=46.64, SSIM=0.9953, LPIPS=0.0850</td>
</tr>
<tr>
<td width="50%"><img src="assets/rhd_results/warming_workpieces_top1_00015.png" width="176"> <img src="assets/rhd_results_pseudo/warming_workpieces_top1_00015.png" width="176"><br><sub>gray / RHD pseudo</sub><br><b>warming_workpieces</b> / view 00015.png<br>t=0.5643, PSNR=47.81, SSIM=0.9955, LPIPS=0.0559</td>
<td width="50%"><img src="assets/rhd_results/warming_workpieces_top2_00008.png" width="176"> <img src="assets/rhd_results_pseudo/warming_workpieces_top2_00008.png" width="176"><br><sub>gray / RHD pseudo</sub><br><b>warming_workpieces</b> / view 00008.png<br>t=0.3236, PSNR=46.97, SSIM=0.9958, LPIPS=0.0379</td>
</tr>
</table>

## Dataset

RHD scenes are expected to be arranged as COLMAP-style thermal folders:

```text
dataset/RHD/
  cooling_bench/
    thermal/
      images/
      sparse/0/
      info.json
  ...
```

With `--eval`, the thermal split is created by holding out centered temporal segments from each quarter of the sequence, matching the current code path.

## Environment

```bash
conda env create -f environment.yml
conda activate thermal3dgs
```

If you already have a compatible PyTorch/CUDA environment, install only the missing packages and build the Gaussian rasterization submodules as usual.

## Training

Train one scene:

```bash
python train.py -s dataset/RHD/cooling_bench/thermal -m output/RHD_temporal_physir_splat/cooling_bench --eval
```

Run the RHD thermal split:

```bash
bash scripts/run_rhd_temporal_physir.sh
```

## Rendering And Evaluation

```bash
python render.py -m output/RHD_temporal_physir_splat/cooling_bench --iteration 30000 --skip_train
python metrics.py -m output/RHD_temporal_physir_splat/cooling_bench
```

The batch script trains, renders, evaluates, and writes a CSV summary:

```text
output/RHD_temporal_physir_splat/metrics_summary.csv
```

## Acknowledgements

This codebase builds on 3D Gaussian Splatting infrastructure and is inspired by recent thermal 3DGS work. We thank the authors of:

```bibtex
@article{wang2026etgs,
  title={ETGS: Explicit Thermodynamics Gaussian Splatting for Dynamic Thermal Reconstruction},
  author={Wang, Zhongwen and Ling, Han and Zhang, Weihao and Sun, Yinghui and Sun, Quansen},
  journal={International Conference on Learning Representations},
  year={2026}
}

@inproceedings{luthermalgaussian,
  title={ThermalGaussian: Thermal 3D Gaussian Splatting},
  author={Lu, Rongfeng and Chen, Hangyu and Zhu, Zunjie and Qin, Yuhang and Lu, Ming and Yan, Chenggang and others},
  booktitle={International Conference on Learning Representations},
  year={2025}
}

@inproceedings{chen2024thermal3d,
  title={Thermal3D-GS: Physics-Induced 3D Gaussians for Thermal Infrared Novel-View Synthesis},
  author={Chen, Qian and Shu, Shihao and Bai, Xiangzhi},
  booktitle={European Conference on Computer Vision},
  pages={253--269},
  year={2024}
}

@article{chen2026thermal3dpami,
  title={Thermal3D-GS: Physics-Induced 3D Gaussians for Thermal Infrared Novel-View Synthesis With a Large-Scale Dataset},
  author={Chen, Qian and Shu, Shihao and Sun, Heng and Chen, Junzhang and Bai, Xiangzhi},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={48},
  number={6},
  pages={6962--6979},
  year={2026},
  doi={10.1109/TPAMI.2026.3663966}
}

@inproceedings{nam2025veta,
  title={Veta-GS: View-Dependent Deformable 3D Gaussian Splatting for Thermal Infrared Novel-View Synthesis},
  author={Nam, Myeongseok and Park, Wongi and Kim, Minsol and Hur, Hyejin and Lee, Soomok},
  booktitle={IEEE International Conference on Image Processing},
  pages={965--970},
  year={2025}
}

@article{kerbl2023gaussians,
  title={3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  author={Kerbl, Bernhard and Kopanas, Georgios and Leimkuehler, Thomas and Drettakis, George},
  journal={ACM Transactions on Graphics},
  volume={42},
  number={4},
  year={2023}
}
```

## License

This repository contains components derived from third-party research code. Keep the accompanying license files and source notices when redistributing or publishing modified versions.

## Citation

If you find this repository useful, please cite PhysIR-Splat:

```bibtex
@inproceedings{gao2026physirsplat,
  title={PhysIR-Splat: Physically Consistent Thermal Infrared Radiative Transfer in 3D Gaussian Splatting},
  author={Gao, Jingyuan and Zhang, Qiming and Gao, Fei and Zhang, Mingjin},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```
