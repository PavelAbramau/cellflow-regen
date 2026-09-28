# Notes: response-space family structure (07 Part II) and receptor competence (09)

Written 2026-09-27. The model is `models/06_baseline_CellFlow.pkl` throughout. Notebook 01 is the GPU inventory and
saved no model; 06 is the only trained model (20 cytokines, 4 donors, IL-15 held out for Donor1/2/12).

---

## 1. Do cytokine *responses* have receptor-family structure? (07 Part II)

Notebook: [notebooks/07_condition_embedding.ipynb](notebooks/07_condition_embedding.ipynb), Part II.
Figure: `models/07_response_vs_embedding_structure.png`.

### Setup
- **Response vector.** For each donor × cytokine × cell type: pseudobulk mean log1p(CP10k) minus the same donor's PBS
  mean for that cell type, over the 2,000 HVGs.
  - Groups need ≥ 20 cells on both sides.
  - Deltas are averaged over donors.
  - Each cytokine's vector is the concatenation over the 14 cell types that have a delta for every cytokine. HSPC, ILC,
    Plasmablast and pDC are dropped.
- **Data.** The full HF dataset (`pbmc_parse.h5ad`, 90 cytokines, 12 donors). Group means are cached at
  `/fast/AG_Bunina/Pavel_1/data/cellflow/pbmc_parse_pseudobulk_log1p_means.h5ad`.
- **Tests.** Same machinery as Part I:
  - cosine silhouette with `cytokine_family` labels vs 100 label permutations;
  - within- vs between-group cosine distance for 4 shared-receptor groups, with a percentile against random groups of
    the same size.
- **Labels for the 90-cytokine test.** Families with ≥ 2 members, excluding the catch-all bins "Other" and
  "Interleukin - non-classified", which leaves 77 cytokines in 10 families.
- **Reliability.** Split-half correlation of each response vector (donors 1–6 vs 7–12).

### Results
**Half the cytokines barely respond.** Only 40 of 90 have a reproducible response (split-half r ≥ 0.5). Among the least
reproducible are SCF, IL-36Ra, IL-17E, GDNF, LIF, TGF-β1, IL-17F and IL-23.

**Family silhouette vs 100 permutations:**

| space | 14 trained cytokines (4 families) | 77 cytokines (10 families) | 40 reliable responders |
|---|---|---|---|
| response (data) | −0.021, p = 0.10 | −0.118 (null −0.18), p = 0.01 | −0.058 (null −0.21), p = 0.01 |
| ESM2 raw | 0.097, p = 0.04 | −0.142, p = 0.01 | −0.211, p = 0.04 |
| ESM2 centred | 0.139, p = 0.02 | −0.020, p = 0.01 | −0.080, p = 0.01 |
| learned (donor mean) | −0.117, p = 0.17 | −0.191, p = 0.01 | −0.120, p = 0.01 |

(p = 0.01 is the smallest value possible with 100 permutations.)

**Per-family silhouette, 77 cytokines.** In the data, only the **interferons** cohere (+0.24); every other family is
negative. ESM2 groups sequence families (complement, common β chain, IL-17). The learned embedding groups complement,
IL-17 and IL-6/IL-12. Neither embedding groups the interferons (−0.30 / −0.42).

**Shared-receptor groups** (percentile of within-group distance among random groups; low = tight; all spaces scored on
the same 90 cytokines):

| group | response (data) | ESM2 raw | ESM2 centred | learned |
|---|---|---|---|---|
| IFNAR (IFN-α1, -β, -ε, -ω) | **0.1** | 0.2 | 0.0 | 14.4 |
| CSF1R (M-CSF, IL-34) | 21.8 | 9.8 | 3.6 | 26.7 |
| gp130 (IL-6, IL-11, LIF, OSM, CT-1) | 84.2 | 0.4 | 0.1 | 1.7 |
| IL17RA/RC (IL-17A, IL-17F) | 23.1 | 0.2 | 0.3 | 0.1 |

- Split-half reliability of the group members:
  - IFNs: 0.76–0.94
  - M-CSF 0.79, IL-34 0.65
  - IL-6 0.71, CT-1 0.71, OSM 0.57, IL-11 0.29, LIF 0.23
  - IL-17A 0.33, IL-17F 0.24
- The per-cell-type silhouette passes permutations mainly in myeloid cells (cDC, CD16/CD14 monocytes) and in NK
  CD56bright. Naive CD4 T cells show no family structure (p = 0.20).

### Answer
**Neither of the two expected outcomes appeared cleanly.**
- PBMC responses cluster by family only weakly.
- The one shared-receptor group that is tight in the data is IFNAR. ESM2 also groups it, but the learned embedding
  loosens it. That comparison is unfair to the model, though: only IFN-ε was in training.
- gp130 and IL-17A/F are tight in ESM2 and in the model, but the cells don't respond alike. The shared β chain doesn't
  decide a blood-cell response; the ligand-specific α chains do, and IL-11 and LIF barely respond.
- So the receptor premise holds in PBMCs only where the full receptor is expressed on blood cells.
- A more accurate motivating figure is: *the embeddings group by sequence family; the cells group by which receptors
  they actually carry; the two agree only for IFNAR.*

---

## 2. Receptor competence (09)

Notebook: [notebooks/09_receptor_competence.ipynb](notebooks/09_receptor_competence.ipynb).
Outputs: `models/09_response_per_donor.csv`, `models/09_receptor_vs_response.csv`, `models/09_model_vs_observed.csv`.

**Finding (the top of the notebook):** The seven receptor-poor ligands produce no PBMC response above well-to-well
noise while most receptor-rich ligands respond in nearly every cell type, and the 06 model predicts near-zero for the
receptor-poor ligands only because it predicts near-zero for *every* unseen ligand, including IFN-γ, whose observed
response is 4× the noise floor.

### Data decisions
- **Responses.** The full HF dataset: 1.9M cells (13 ligands + PBS, 12 donors). 8 of the 13 ligands are not in
  `pbmc_subset.h5ad`.
- **Receptor expression.** The HF matrix (2,000 HVGs, no raw counts) contains only 4 of the 19 receptor genes. Instead
  I used Parse's PBS pseudobulks: per-donor raw-count sums over all 40,352 genes.
  - The 24 PBS files were pulled out of the remote `pseudobulks_20260203.zip` with HTTP range requests: 330 MB
    transferred, not the 44 GB h5ad.
  - Location: `/fast/AG_Bunina/Pavel_1/data/cellflow/parse_pseudobulk_PBS/`. Helper:
    [scripts/http_range_file.py](scripts/http_range_file.py).
  - Licence: CC BY-NC 4.0. Parse's download page asks users to fill in a form; I used the direct S3 links.
- **GFRA4** is not in Parse's gene annotation, so PSPN is scored on RET alone.
- **Cell-type mapping.** The pseudobulk labels are Parse's release-time annotation, mapped to the HF labels by name.
  PBS cell counts agree closely for monocytes, DCs, B, NK and HSPC (per-donor count correlation ≥ 0.9). They agree less
  well for CD4 naive/memory, Treg and MAIT (0.71–0.98) and poorly for ILC (0.14).

### Method
- Per donor × ligand × cell type, both sides are subsampled to the same n = min(n_ligand, n_PBS, 500), with n ≥ 30.
- **DE:** Wilcoxon, BH FDR < 0.05, |log2FC| > 0.5.
- **Energy distance:** `cellflow.metrics.compute_e_distance` in 50 PCs fitted on a balanced subsample.
- Everything is summarised as the median over donors.
- **Receptor score:** mean log1p(CPM) over the listed subunits, averaged over donors. A sensitivity version uses the
  scarcest subunit.

### Two method issues found along the way
1. **Well confound.** Each cytokine × donor sits in **one** barcoding well (`bc1_well`); PBS spans **6 wells** per
   donor. So the naive PBS-vs-PBS null (random halves) is too low:

   | null | energy distance | DE genes |
   |---|---|---|
   | random PBS halves | 1.74 | 0.0 |
   | one PBS well vs the other PBS wells | 9.17 | 1.75 |

   Against the random null, every receptor-poor ligand looked like a weak responder in 14–16 cell types. Against the
   well null, they don't respond at all. **Any ligand-vs-PBS comparison in this dataset needs the well-level null.**
2. **Energy-distance bias.** CellFlow's `compute_e_distance` includes the zero self-distances (a V-statistic), so it is
   biased upwards by roughly (σ_X + σ_Y)/n. Floors computed at different n aren't comparable, so in part 4 every
   comparison within a donor × cell type now uses the same n.

### Results: observed responses (well floor)
| ligand | group | median DE | median e-dist | cell types > 2× well floor (e-dist) | cell types > floor (DE) |
|---|---|---|---|---|---|
| EGF | poor | 2.0 | 9.5 | 0/16 | 0 |
| FGF-beta | poor | 2.0 | 10.5 | 1/16 | 0 |
| HGF | poor | 0.5 | 7.2 | 0/15 | 0 |
| IGF-1 | poor | 3.5 | 9.9 | 1/15 | 1 |
| VEGF | poor | 2.0 | 7.0 | 0/15 | 0 |
| GDNF | poor | 1.0 | 6.2 | 0/16 | 0 |
| PSPN | poor | 1.0 | 7.4 | 0/15 | 0 |
| GM-CSF | rich | 8.0 | 21.8 | 9/17 | 7 |
| IFN-γ | rich | 35.0 | 40.8 | 16/16 | 13 |
| IL-15 | rich | 150.3 | 165.9 | 16/16 | 16 |
| IL-2 | rich | 94.0 | 103.1 | 16/16 | 15 |
| IL-4 | rich | 46.3 | 55.0 | 14/16 | 14 |
| TNF-α | rich | 2.0 | 6.8 | 2/17 | 1 |

"> floor (DE)" means more than 2 × well floor + 5 DE genes.

- **TNF-α is effectively null** despite high TNFRSF1A/1B.
- **IGF-1 is null** despite IGF1R being abundant in T cells (log1p CPM 5–7).
- So receptor mRNA is necessary but not sufficient.

### Results: receptor expression vs response (206 ligand × cell-type pairs)
| pairs | Spearman ρ, DE | Spearman ρ, e-dist |
|---|---|---|
| all | 0.55 (p < 10⁻¹⁵) | 0.53 |
| receptor-rich only | 0.34 (p = 6×10⁻⁴) | 0.22 (p = 0.03) |
| receptor-poor only | 0.22 (p = 0.02) | 0.24 (p = 0.01) |

Much of the pooled correlation comes from the two-group split. The scarcest-subunit receptor score gives the same
picture (all-pairs ρ = 0.53 / 0.50).

### Results: model predictions (part 4)
Measured in the model's 100-PC space, for its 4 donors, with every comparison at matched n. The well floor there is
21.3, the random floor 13.1.

| ligand | status in model | observed / well floor | predicted / well floor | predicted / observed |
|---|---|---|---|---|
| EGF | unseen | 1.10 | 0.77 | 0.71 |
| FGF-beta | trained | 1.11 | 0.79 | 0.71 |
| HGF | unseen | 0.92 | 0.73 | 0.79 |
| IGF-1 | unseen | 1.35 | 0.86 | 0.63 |
| VEGF | unseen | 0.90 | 0.78 | 0.87 |
| GDNF | unseen | 1.01 | 0.90 | 0.90 |
| PSPN | unseen | 0.99 | 0.88 | 0.89 |
| GM-CSF | trained | 2.49 | 2.07 | 0.83 |
| IL-15 | trained (Donor8 only) | 9.71 | 5.85 | 0.60 |
| TNF-α | unseen | 0.97 | 1.38 | 1.43 |
| **IFN-γ** | **unseen** | **4.03** | **0.91** | **0.23** |
| IL-4 | trained | 5.06 | 6.61 | 1.31 |
| IL-2 | trained | 5.92 | 4.84 | 0.82 |

### Answer to part 4
- The model does **not** predict a substantial effect for the receptor-poor ligands, which matches the observed null.
- It gets that right for the wrong reason. It predicts a near-floor response for every unseen ligand, including IFN-γ,
  whose receptor is abundant and whose observed response is among the largest.
- From the ESM2 vector alone, the model can't separate "receptor present on these cells" from "receptor absent".
- Receptor competence is the dominant factor in whether PBMCs respond at all, and the current model has no access to
  it.

---

## Status of other work
- **08 (family one-hot / shuffled-ESM2 arms):**
  - Smoke test passed. Arm A reproduces 06: R² identical to 4 decimals, e-distance within 0.03.
  - The full run (job 3375645, arms B and C at 500k iterations each) is still training; results and conclusion pending.
  - Caveat to keep in mind: IL-15 is in training for Donor8, so the shuffled arm (C) can use its vector as an arbitrary
    ID. C ≈ A would therefore not show that the representation is ignored.
- **07 Part I:** the learned condition embedding has not rediscovered receptor structure. Where receptor groups are
  tight in it, the structure is inherited from ESM2; see the Part I conclusion in the notebook.
