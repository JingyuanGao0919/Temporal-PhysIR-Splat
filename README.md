# Temporal PhysIR-Splat

## Method Overview

Temporal PhysIR-Splat extends **PhysIR-Splat: Physically Consistent Thermal Infrared Radiative Transfer in 3D Gaussian Splatting** to dynamic thermal-image synthesis. Given scene geometry, target cameras, and timestamps, it represents a scene with 3D Gaussian primitives and models a continuous, time-varying thermal field. Gaussians assigned to the same material group share a time-invariant effective in-band emissivity, while their thermal states evolve over time.

The infrared image-formation chain combines thermal self-emission and reflected ambient radiation, accounts for atmospheric transmission, composites the Gaussian contributions according to visibility, and maps the resulting radiance to the dataset's temperature encoding. A jointly trained image-space residual complements the physical rendering. The model is optimized from training thermal observations and queried directly at future timestamps, without test-time optimization.

## CFE-20 Evaluation Protocol

We evaluate future thermal-image synthesis given scene geometry and target cameras on ten RHD scenes introduced by ETGS (Wang et al., 2026), using **Chronological Future Extrapolation (CFE-20, also denoted E20)**. For each scene, retain only frames with both a thermal image and a released COLMAP pose. Parse image filenames as `HHMMSSmmm` timestamps and sort chronologically. For `N` valid frames, set `n = floor(0.8N)`: frames `0, ..., n-1` form the training set and frames `n, ..., N-1` form the test set. The split is based on frame count, with no random shuffling or temporal subsampling.

| RHD scene | Training images | Test images |
|---|---:|---:|
| cooling_bench | 176 | 44 |
| cooling_checkboard | 174 | 44 |
| cooling_dumbbels | 168 | 42 |
| cooling_ebike | 252 | 63 |
| heat_transfer | 200 | 50 |
| heating_workpieces | 179 | 45 |
| warming_bottles | 167 | 42 |
| warming_cups | 179 | 45 |
| warming_peaches | 203 | 51 |
| warming_workpieces | 190 | 48 |
| **Total** | **1,888** | **474** |

All methods use **512 × 410** images and share the split, released COLMAP camera intrinsics and poses, initial point clouds, fixed temperature encoding, and ambient-temperature metadata. Geometry and target cameras are provided as inputs. The image `cooling_bench/160634031.png` is excluded because it has no released COLMAP pose.

Time is measured in seconds from the first valid frame, preserving the recorded sampling intervals. Where normalized time is required, use `u_i = (t_i - t_0) / (t_(n-1) - t_0)`. Future test timestamps consequently exceed 1; do not clamp them or normalize by the full sequence duration.

## Acknowledgements

This work builds on 3D Gaussian Splatting infrastructure and is inspired by recent thermal 3DGS work. We thank the authors of:

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


