# DATA — PBMC cytokine subset

**File:** `/fast/AG_Bunina/Pavel_1/data/cellflow/pbmc_subset.h5ad` (also reachable as `/fast/AG_Bunina/Pavel_1/cellflow-regen/data/pbmc_subset.h5ad`)
**Size:** 331,850,186 bytes (0.33 GB) · **sha256:** `61883f332841d2af7becd91d4d650d7ef24f329e5f0b33df6dd8ca617509edc1`
**Shape:** 1,114,081 cells × 2,000 genes · all 18 cell types · written with `compression="gzip"`
**Built by:** [`notebooks/05_build_subset.ipynb`](notebooks/05_build_subset.ipynb) (kernel `cellflow`)

## Source
* `cellflow.datasets.pbmc_cytokines()` (cellflow-tools 0.0.9, anndata 0.12.19) →
  HF dataset `theislab/cellflow-datasets`, file `pbmc_parse.h5ad`, revision `4b235c9a1e0073c95211c6ef6257db184fc83f71`
* 13,101,592,422 bytes, LFS sha256 `c72671b36248b82c559211a28fbf009f4f077e3b2097f9ba6a04d27d27edd4c9`
* 9,697,974 cells × 2,000 HVGs; cached at `/fast/AG_Bunina/Pavel_1/data/cellflow/hf_cache/hub/datasets--theislab--cellflow-datasets/snapshots/4b235c9a1e0073c95211c6ef6257db184fc83f71/pbmc_parse.h5ad` (`HF_HOME=/fast/AG_Bunina/Pavel_1/data/cellflow/hf_cache`)

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

## Regenerate
```bash
source /fast/AG_Bunina/Pavel_1/miniforge3/bin/activate /fast/AG_Bunina/Pavel_1/miniforge3/envs/cellflow
cd /fast/AG_Bunina/Pavel_1/cellflow-regen/notebooks
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=cellflow 05_build_subset.ipynb
sha256sum /fast/AG_Bunina/Pavel_1/data/cellflow/pbmc_subset.h5ad   # should match the value above
```
Environment: see `/fast/AG_Bunina/Pavel_1/cellflow-regen/requirements.lock.txt` and `notebooks/02_env_setup.ipynb`.
