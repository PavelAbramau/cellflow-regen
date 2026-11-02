# cellflow-regen

**Does the way we represent a cytokine tell a generative model how cells will respond to it?**

This repo tests that question on the Parse PBMC cytokine atlas (about 9.7 M cells, 90 cytokines, 12 donors, 18 cell types) using [CellFlow](https://github.com/theislab/cellflow). CellFlow conditions on a vector for each perturbation. By default that vector is an ESM2 protein-language-model embedding. We ask whether it should be.

## What we found so far

- **Half the cytokines barely do anything.** Only 40–46 of 90 give a reproducible response (split-half r ≥ 0.5). The rest are noise, which caps what any model can learn.
- **ESM2 does not predict response geometry.** Spearman ρ between ESM2 distances and response distances is about 0.04, no better than random vectors.
- **Receptor binding and cytokine family carry weak but real signal** (ρ ≈ 0.11 and 0.13, Mantel p < 0.001). The gain over ESM2 is directional and not statistically demonstrated. Binding may be largely the family signal in disguise.
- Only the interferons form a tight family in response space. Most other families do not.

Details are in [notes.md](notes.md) and [scripts/build/nb22_verdict.md](scripts/build/nb22_verdict.md). A full write-up is in [reports/cellflow_experiments_report.pdf](reports/cellflow_experiments_report.pdf).

## Layout

| Path | What's inside |
|---|---|
| `notebooks/` | Numbered analysis, in order: 01–09 baseline CellFlow and condition embeddings; 20 response matrix; 21 receptor layer; 22 representation screen; 30–32 training sets and baselines; 40–41 zero-shot pilot |
| `scripts/` | Pseudobulking, UniProt fetch, report builder, SLURM `*.sbatch` launchers. `scripts/build/` holds the generators for the numbered notebooks |
| `cellflow_fork/` | Our modified CellFlow (arms for alternative condition representations) |
| `tests/` | Unit tests for the fork |
| `models/` | Metrics (CSV) and figures (PNG) only. Checkpoints are not tracked |
| `reports/` | Generated PDF report |
| [DATA.md](DATA.md) · [RECEPTORS.md](RECEPTORS.md) · [SETUP.md](SETUP.md) · [HARDWARE.md](HARDWARE.md) | Dataset inventory, receptor/binding layer, environment, GPU notes |

## Reproducing

```bash
conda activate /users/scratch1/pavelabramau/envs/cellflow   # python 3.11, jax[cuda12], cellflow-tools
pip install -r requirements.lock.txt                         # or build a fresh env
ln -s <big-storage>/cellflow data                            # data/ is not tracked
sbatch scripts/run_nb.sbatch notebooks/20_response_matrix.ipynb
```

Notebooks run through papermill on SLURM (A100 80 GB for training, CPU for notebooks 20–22). JAX first use on a GPU node should be checked with `python -c "import jax; print(jax.devices())"`; see [SETUP.md](SETUP.md).

## Data and licences

Not in git: the 13 GB `pbmc_parse.h5ad`, pseudobulk caches, and model checkpoints. The dataset is fetched from the HF hub (`theislab/cellflow-datasets`). The Parse data is **CC BY-NC 4.0**, so non-commercial use only. Receptor annotations come from OmniPath (academic licence).
