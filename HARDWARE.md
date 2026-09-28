# HARDWARE — MDC max-cluster GPU inventory

Collected 2026-09-25 with `sinfo -p gpu -N -o "%N %G %m %t"` and
`sinfo -p interactive -N -o "%N %G %m %t"` (run on `max-login` via ssh;
the OOD session container has no SLURM client). Raw output:
[notebooks/01_hardware_inventory.ipynb](notebooks/01_hardware_inventory.ipynb).

VRAM is the per-GPU `gpu_memory` GRES; host RAM is the node's `RealMemory`.

## GPU node types

| GPU model | VRAM / GPU | GPUs / node | Host RAM / node | Nodes | In `gpu` | In `interactive` |
|---|---|---|---|---|---|---|
| NVIDIA Tesla T4 | 15 GB | 1 | ~991 GB (1014717 MB) | max001–max010 (10) | yes | yes |
| NVIDIA Tesla V100 (16G) | 16 GB | 8 | ~487 GB (498946 MB) | maxg01 (1) | yes | yes |
| NVIDIA Tesla V100 (32G) | 32 GB | 4 | ~171 GB (175135 MB) | maxg05–maxg08 (4; maxg05 drained) | yes | yes |
| NVIDIA A40 | 45 GB | 3 | ~739 GB (756270 MB) | max020–max029 (10; max027 only in `interactive`) | yes | yes |
| NVIDIA A40 | 45 GB | 8 | ~1.46 TB (1529708 MB) | maxg11–maxg18, maxg21–maxg26 (14) | yes | yes |
| NVIDIA H100 | 80 GB | 8 | ~991 GB (~1014630 MB) | maxg09, maxg10, maxg20 (3) | yes | yes |
| NVIDIA H200 | 140 GB | 8 | ~991 GB (1014630 MB) | maxg19, maxg29 (2) | yes | yes |
| NVIDIA B300 | 268 GB | 8 | ~2.0 TB (2046957 MB) | maxg03, maxg04 (2) | yes | yes |

Totals (from partition TRES): `gpu` partition 229 GPUs (139 A40, 24 H100, 16 H200,
16 B300, 24 V100, 10 T4); `interactive` 232 GPUs.

## Is there >=40 GB VRAM?

**Yes.** A40 (45 GB), H100 (80 GB), H200 (140 GB) and B300 (268 GB) nodes all have >=40 GB
per GPU, in both partitions. The A40s are the most plentiful (24 nodes, 139 GPUs,
several idle at inventory time), so the plan does not need to change.

## Partition limits that matter

| Partition | Max walltime | Max mem / node |
|---|---|---|
| `interactive` | 2 days | 100 GB (`MaxMemPerNode=102400`) |
| `gpu` | 7 days | node RealMemory |

## Caveats

* A bare `--gres=gpu:1` can be scheduled on a **15 GB T4** or a 16 GB V100. For training,
  request the model explicitly, e.g. `--gres=gpu:a40:1` (>=45 GB) or `--gres=gpu:h100:1`.
* B300 (compute capability 10.0) needs a recent CUDA 12.8+ JAX build; A40/H100 work with any
  current `jax[cuda12]`.
* The OOD Jupyter session node (max031, `normal` partition) has **no GPU**.
