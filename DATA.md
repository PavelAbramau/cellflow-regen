# DATA — full PBMC cytokine dataset (`cellflow.datasets.pbmc_cytokines()`)

Inventory 2026-11-01; read via `anndata.read_h5ad(backed="r")` / h5py, no X load into memory.
File: `data/cellflow/hf_cache/hub/datasets--theislab--cellflow-datasets/snapshots/4b235c9a…/pbmc_parse.h5ad`
(13,101,592,422 bytes; same size as recorded sha256 `c72671b3…d4c9`; hash not re-checked). Licence of the Parse data: CC BY-NC 4.0.

* **Shape**: 9,697,974 cells × 2,000 genes (HVGs, `var.highly_variable` all True); X is CSR float32, 964.5 M non-zeros; no layers, no raw.
* **obs columns (21)**: sample, species, gene_count, tscp_count, mread_count, bc1_wind, bc2_wind, bc3_wind, bc1_well, bc2_well, bc3_well,
  log1p_n_genes_by_counts, log1p_total_counts, total_counts_MT, pct_counts_MT, log1p_total_counts_MT, donor, cytokine, treatment, cell_type, cytokine_family
* **var columns**: n_cells, highly_variable, means, dispersions, dispersions_norm
* **uns keys**: donor_embeddings, esm2_embeddings, hvg, log1p (empty dict)
* **Donors (12)**: Donor1–Donor12. **Cytokines (91)** = 90 cytokines + `PBS` control (treatment: PBS / cytokine).
  **cell_type (18)**: B Intermediate/Memory, B Naive, CD14 Mono, CD16 Mono, CD4 Memory, CD4 Naive, CD8 Memory, CD8 Naive, HSPC, ILC, MAIT, NK, NK CD56bright, NKT, Plasmablast, Treg, cDC, pDC.
  **cytokine_family (15 incl. NaN)**: Chemokine, Common β chain, Common γ chain/IL-13/TSLP, Complement, Growth factor, IL-1, IL-10, IL-10 (IL-20 subfamily), IL-17, IL-6/IL-12, Interferon, Interleukin - non-classified, Other, TNF, nan (PBS).

## uns['esm2_embeddings']
* Exists: **93 entries**, each a **2560-d** vector (ESM2).
* **Not L2-normalised**: norms min 9.54, max 14.78, mean 12.84. Normalise explicitly if cosine/L2 conventions matter.
* **Cytokines in obs missing from the dict: `PBS`** (control, expected). Dict extras not in obs: `EBI3`, `LT-alpha`, `LT-beta` (subunits of IL-35 / LT-alpha1-beta2 etc.).

## Normalisation decision (BEFORE any normalize_total/log1p)
* `X.max()` over all 964.5 M stored values = **1897.0**; all sampled values are integers ≥ 1 (min 1.0, 100 % integer); `uns['log1p']` is empty.
* **Decision: X is raw UMI counts, NOT log-transformed.** Apply `normalize_total(target_sum=1e4)` then `log1p` (log1p(CP10k)).
  Caveat: X holds only the 2,000 HVGs, so CP10k on X-row sums would be relative to HVG counts; per-cell totals over all genes are available in `obs['tscp_count']`/`gene_count` — use a stated choice and keep it consistent across notebooks 20–22.
  (Row sums of `expm1(X)` blow up/overflow, confirming X is not log space.)
* No training was run.

## Recipe
1. **Donors (4):** `Donor8` + the 3 other donors with the most PBS cells in the full data (ties by name)
   → `['Donor8', 'Donor1', 'Donor12', 'Donor2']`
2. **Cytokines (20):** `IL-15` + all other cytokines in its family (`Common γ chain/IL-13/TSLP`): `['IL-13', 'IL-2', 'IL-21', 'IL-4', 'IL-7', 'IL-9', 'TSLP']`;
   remaining 12 slots drawn with `np.random.default_rng(0).choice(pool, size=12, replace=False)`
   where `pool` = sorted non-PBS cytokines not already chosen
   → `['IL-15', 'IL-13', 'IL-2', 'IL-21', 'IL-4', 'IL-7', 'IL-9', 'TSLP', 'ADSF', 'BAFF', 'C5a', 'FGF-beta', 'GM-CSF', 'IFN-epsilon', 'IL-17B', 'IL-20', 'IL-31', 'IL-6', 'LT-alpha2-beta1', 'SCF']`
3. **Controls:** `PBS` for all 4 donors.
4. **Cells:** `donor ∈ DONORS and cytokine ∈ CYTOKINES ∪ {PBS}`, all cell types. `obs`, `var`, `uns` kept
   (unused categories removed); recipe stored in `uns['subset_recipe']`.

Checks passed: PBS and IL-15 present for every donor; all 20 cytokines have an entry in `uns['esm2_embeddings']`
(2560-d ESM2 vectors); all donors in `uns['donor_embeddings']`.

## Cells per donor
| donor | cells |
|---|---|
| Donor8 | 187,549 |
| Donor1 | 362,067 |
| Donor12 | 309,565 |
| Donor2 | 254,900 |

## Cells per cytokine
| cytokine | cells |
|---|---|
| IL-15 | 41,791 |
| IL-13 | 47,827 |
| IL-2 | 40,176 |
| IL-21 | 51,808 |
| IL-4 | 50,125 |
| IL-7 | 48,589 |
| IL-9 | 47,897 |
| TSLP | 44,461 |
| ADSF | 46,844 |
| BAFF | 26,114 |
| C5a | 31,363 |
| FGF-beta | 35,984 |
| GM-CSF | 44,008 |
| IFN-epsilon | 42,943 |
| IL-17B | 33,205 |
| IL-20 | 40,516 |
| IL-31 | 41,885 |
| IL-6 | 37,331 |
| LT-alpha2-beta1 | 32,972 |
| SCF | 53,027 |
| PBS | 275,215 |

## Cells per cytokine × donor
| cytokine | Donor8 | Donor1 | Donor12 | Donor2 | total |
|---|---|---|---|---|---|
| IL-15 | 8,939 | 9,863 | 13,166 | 9,823 | 41,791 |
| IL-13 | 11,671 | 11,890 | 14,840 | 9,426 | 47,827 |
| IL-2 | 5,817 | 15,777 | 11,535 | 7,047 | 40,176 |
| IL-21 | 9,890 | 19,986 | 12,921 | 9,011 | 51,808 |
| IL-4 | 5,602 | 16,190 | 15,731 | 12,602 | 50,125 |
| IL-7 | 8,721 | 15,891 | 15,873 | 8,104 | 48,589 |
| IL-9 | 11,218 | 7,402 | 15,622 | 13,655 | 47,897 |
| TSLP | 8,583 | 8,279 | 16,321 | 11,278 | 44,461 |
| ADSF | 7,966 | 15,211 | 11,641 | 12,026 | 46,844 |
| BAFF | 4,541 | 9,372 | 8,234 | 3,967 | 26,114 |
| C5a | 6,439 | 9,745 | 7,433 | 7,746 | 31,363 |
| FGF-beta | 4,131 | 14,008 | 7,471 | 10,374 | 35,984 |
| GM-CSF | 4,828 | 19,208 | 10,526 | 9,446 | 44,008 |
| IFN-epsilon | 6,633 | 17,454 | 11,793 | 7,063 | 42,943 |
| IL-17B | 5,038 | 14,153 | 8,832 | 5,182 | 33,205 |
| IL-20 | 6,301 | 13,140 | 11,606 | 9,469 | 40,516 |
| IL-31 | 7,400 | 12,831 | 12,889 | 8,765 | 41,885 |
| IL-6 | 6,344 | 11,596 | 9,747 | 9,644 | 37,331 |
| LT-alpha2-beta1 | 5,061 | 13,929 | 8,533 | 5,449 | 32,972 |
| SCF | 9,362 | 15,690 | 13,672 | 14,303 | 53,027 |
| PBS | 43,064 | 90,452 | 71,179 | 70,520 | 275,215 |
