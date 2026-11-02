# HARDWARE — GPU inventory (cluster `trytonp-ap`, Cyfronet/TASK-KDM Slurm)

Collected 2026-11-01 from the login node `trytonp-ap` (24 CPUs, 376 GB RAM, no GPU).
Command: `sinfo -p <partition> -N -o "%N %G %m %c %t %l"` plus `scontrol show partition/node`.
Replaces the earlier inventory for the MDC max-cluster (still in git history, commit 705be78).

| Partition | GPU (Slurm GRES) | GPUs/node | Nodes | Host RAM/node | CPUs/node | Max walltime |
|---|---|---|---|---|---|---|
| `gpu-a100-80gb` | A100 80GB (`a100_80gb`) | 1 | clgk01–24 (24) | 72,000 MB | 42 | 3 days |
| `gpu-h100` | H100 (`h100`) | 4 | 16 (`agj2a*`) | 754,000 MB | 128 | 3 days |
| `gpu-a100` | A100 (`a100`) | 1/2/4 | akiv09,10,11,13 (4) | 252,000–512,000 MB | 4–8 | 3 days |

* **VRAM**: Slurm reports only the GRES model name. 80 GB for `a100_80gb` follows from the name; H100 is
  presumably 80 GB and `a100` (akiv) presumably 40 GB — **not verifiable without a GPU shell (`nvidia-smi`)**; confirm on first GPU job.
* **State at inventory**: `gpu-a100-80gb` 23/24 allocated, 1 reserved; `gpu-a100` all 4 nodes down; `gpu-h100`
  13 allocated, 3 reserved, 2 down. GPUs are heavily contended. A test `srun` on `gpu-a100-80gb` queued and was cancelled.
* **Host RAM warning**: `gpu-a100-80gb` nodes have only 72 GB RAM / 42 cores for a single GPU.
* Partition access: `AllowAccounts=ALL`, `AllowQos=ALL`, no `MaxMemPerNode` limit on the GPU partitions.
* Other partitions (CPU, notebooks 20–22): `interact` / `gui` 8 h, `batch`/`plgrid` 3 d, `plgrid-long` 7 d, `test` 15 min.

## Per-user QOS / association caps — NOT OBTAINED
`sacctmgr show assoc user=$USER` fails: `slurmdbgt1.gt.kdm.task.gda.pl:7031: No route to host`
(accounting DB unreachable from this host). Known instead: `AccountingStorageEnforce=associations,limits`
(so limits exist but cannot be read), account `users`, FairShare 0.046 (`sshare -U`). Re-run `sacctmgr` when the DB is reachable.

## Storage
* `/users/kdm` (home/project root): NFS, 12 TB, 79 % used.
* `/users/scratch1/pavelabramau` (Lustre `kdmscr`, 769 TB, 83 % used): large data, env, caches.
