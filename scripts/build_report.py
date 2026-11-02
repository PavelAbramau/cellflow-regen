"""Build reports/cellflow_experiments_report.pdf from the notebooks' saved outputs (numbers are read from files or copied from the
notebooks/notes with the source named next to them; nothing is re-computed except simple aggregates of the saved CSVs)."""
import os, sys, json, textwrap
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
ROOT = "/users/kdm/pavelabramau/cellflow-regen"; os.chdir(ROOT)
FIG = sys.argv[1] if len(sys.argv) > 1 else "/tmp/report_figs"; os.makedirs(FIG, exist_ok=True)
fdir = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf")
pdfmetrics.registerFont(TTFont("DV", f"{fdir}/DejaVuSans.ttf")); pdfmetrics.registerFont(TTFont("DVB", f"{fdir}/DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DVI", f"{fdir}/DejaVuSans-Oblique.ttf")); pdfmetrics.registerFont(TTFont("DVM", f"{fdir}/DejaVuSansMono.ttf"))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DVB", italic="DVI", boldItalic="DVB")
S = dict(
 body=ParagraphStyle("b", fontName="DV", fontSize=8.6, leading=11.6, spaceAfter=4),
 small=ParagraphStyle("s", fontName="DV", fontSize=7.4, leading=9.6, textColor=colors.HexColor("#444444"), spaceAfter=3),
 h1=ParagraphStyle("h1", fontName="DVB", fontSize=15, leading=19, spaceBefore=10, spaceAfter=6, textColor=colors.HexColor("#1b3a57")),
 h2=ParagraphStyle("h2", fontName="DVB", fontSize=11, leading=14, spaceBefore=8, spaceAfter=3, textColor=colors.HexColor("#1b3a57")),
 h3=ParagraphStyle("h3", fontName="DVB", fontSize=9.2, leading=12, spaceBefore=5, spaceAfter=2),
 title=ParagraphStyle("t", fontName="DVB", fontSize=22, leading=27, spaceAfter=8, textColor=colors.HexColor("#1b3a57")),
 cell=ParagraphStyle("c", fontName="DV", fontSize=7, leading=8.6), cellb=ParagraphStyle("cb", fontName="DVB", fontSize=7, leading=8.6),
 mono=ParagraphStyle("m", fontName="DVM", fontSize=7, leading=8.6), bullet=ParagraphStyle("bl", fontName="DV", fontSize=8.6, leading=11.4, leftIndent=11, bulletIndent=2, spaceAfter=2),
 box=ParagraphStyle("bx", fontName="DV", fontSize=8.6, leading=11.6, backColor=colors.HexColor("#fff4d6"), borderPadding=5, spaceBefore=4, spaceAfter=8))
E = []
def P(t, s="body"): E.append(Paragraph(t, S[s]))
def H1(t): E.append(PageBreak()); P(t, "h1")
def H2(t): P(t, "h2")
def H3(t): P(t, "h3")
def B(items):
    for i in items: E.append(Paragraph(i, S["bullet"], bulletText="•"))
def BOX(t): P(t, "box")
def T(rows, widths=None, header=True, fs=None):
    data = [[Paragraph(str(c), S["cellb"] if (header and r == 0) else S["cell"]) for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0,0), (-1,-1), .3, colors.HexColor("#bbbbbb")), ("VALIGN", (0,0), (-1,-1), "TOP"), ("TOPPADDING", (0,0), (-1,-1), 2), ("BOTTOMPADDING", (0,0), (-1,-1), 2)]
    if header: st.append(("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e6edf3")))
    t.setStyle(TableStyle(st)); E.append(t); E.append(Spacer(1, 5))
def IMG(path, w=17*cm, cap=None):
    from PIL import Image as PI
    iw, ih = PI.open(path).size; E.append(KeepTogether([Image(path, width=w, height=w*ih/iw)] + ([Paragraph(cap, S["small"])] if cap else [])))
def df2rows(df, fmt="{:.3f}", index=True):
    cols = ([df.index.name or ""] if index else []) + [str(c) for c in df.columns]; rows = [cols]
    for i, r in df.iterrows(): rows.append(([str(i)] if index else []) + [fmt.format(v) if isinstance(v, (float, np.floating)) and not np.isnan(v) else ("" if (isinstance(v, float) and np.isnan(v)) else str(v)) for v in r.values])
    return rows

# ------------------------------------------------------------------ derived figures (from saved CSVs)
d06 = pd.read_csv("models/06_baseline_metrics_per_donor.csv"); c06 = pd.read_csv("models/06_baseline_metrics_per_celltype.csv")
m08 = pd.read_csv("models/08_family_vs_esm2_metrics.csv"); r09 = pd.read_csv("models/09_receptor_vs_response.csv")
base = pd.read_csv("data/baselines_train90.csv"); cov = pd.read_csv("data/pbmc_train90_coverage.csv"); rel = pd.read_csv("data/reliability.csv")
rs = pd.read_csv("data/representation_screen.csv"); bvs = pd.read_csv("data/representation_screen_binding_vs_esm2.csv"); ag = pd.read_csv("data/agonist_antagonist_panel.csv")

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2)); t = d06.pivot_table(index="donor", columns="model", values=["e_distance", "r_squared"])
for a, k, yl in zip(ax, ["e_distance", "r_squared"], ["energy distance (lower better)", "R² (higher better)"]):
    t[k].plot.bar(ax=a, color=["#b02f8c", "#9a9a9a"], edgecolor="k", rot=0); a.set_ylabel(yl); a.set_title(k); a.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{FIG}/f06.png", dpi=140); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2)); a8 = m08[m08.cell_type == "all"]; t = a8.pivot_table(index="donor", columns="model", values=["e_distance", "r_squared"])
cols8 = ["#b02f8c", "#3d8aa6", "#e0a030", "#9a9a9a"]
for a, k in zip(ax, ["e_distance", "r_squared"]): t[k].plot.bar(ax=a, color=cols8, edgecolor="k", rot=0); a.set_title(k + " (held-out IL-15, per donor)"); a.legend(fontsize=6)
plt.tight_layout(); plt.savefig(f"{FIG}/f08.png", dpi=140); plt.close()

fig, ax = plt.subplots(figsize=(5.6, 3.8)); cm_ = {"poor": "#c0392b", "rich": "#2471a3"}
for g, s in r09.groupby("group"): ax.scatter(s.receptor_mean, s.log1p_n_de, s=14, c=cm_.get(g, "k"), label=f"receptor-{g}", alpha=.75)
ax.set_xlabel("receptor score (mean log1p CPM over subunits)"); ax.set_ylabel("log1p(# DE genes vs PBS)"); ax.legend(fontsize=7); ax.set_title("09: 206 ligand × cell-type pairs", fontsize=9)
plt.tight_layout(); plt.savefig(f"{FIG}/f09.png", dpi=140); plt.close()

bc = base.pivot_table(index="cell_type", columns="baseline", values="e_distance"); fig, ax = plt.subplots(figsize=(8, 3.6)); bc.plot.bar(ax=ax, color=["#9a9a9a", "#2471a3"], edgecolor="k"); ax.set_ylabel("mean e-distance (100 PCs)"); ax.set_title("31: analytic baselines, leave-one-ligand-out, by cell type", fontsize=9)
plt.xticks(rotation=60, ha="right", fontsize=7); plt.tight_layout(); plt.savefig(f"{FIG}/f31.png", dpi=140); plt.close()

cc = cov.assign(ok=cov.n_cells >= 20).groupby("cell_type").ok.mean(); fig, ax = plt.subplots(figsize=(8, 2.8)); cc.plot.bar(ax=ax, color="#3d8aa6", edgecolor="k"); ax.set_ylabel("fraction of (donor, ligand) groups\nwith ≥ 20 cells"); ax.set_title("30: coverage by cell type (360 groups each)", fontsize=9)
plt.xticks(rotation=60, ha="right", fontsize=7); plt.tight_layout(); plt.savefig(f"{FIG}/f30.png", dpi=140); plt.close()

# ------------------------------------------------------------------ document
P("CellFlow cytokine project — every experiment, notebook by notebook", "title")
P("Methods, data, what was trained, results, statistics, tables and plots · generated 2026-11-02 from the repository <font name='DVM'>/users/kdm/pavelabramau/cellflow-regen</font> (git: one commit, 705be78; most of the work since is uncommitted).", "small")
BOX("<b>How to read this report.</b> Every number is taken from a saved file or notebook output, with the source named. Where a notebook never produced a conclusion, or a job has not run, this report says so instead of filling the gap. "
    "<b>Only one model has ever been trained to completion in this project (notebook 06), plus two more arms in notebook 08.</b> Notebooks 20–22, 30, 31 are CPU-only analyses with no training. Notebooks 32, 40, 41 and the unit tests exist but had <b>not run</b> when this report was written (jobs queued).")

H2("Contents")
T([["Part", "Notebook(s)", "What it is", "Trained a model?"],
   ["0", "01–05", "Cluster GPU inventory, environment, dataset download, 20-cytokine subset", "no"],
   ["1", "06", "CellFlow baseline, IL-15 held out in 3 of 4 donors (donor-transfer)", "<b>yes</b> (500k it)"],
   ["2", "07", "What the learned condition embedding encodes (Part I on the model; Part II on data)", "no (uses 06 model)"],
   ["3", "08", "Family one-hot and shuffled-ESM2 controls vs ESM2", "<b>yes</b> (2 more arms, 500k it each)"],
   ["4", "09", "Receptor competence vs observed response vs the 06 model's predictions", "no"],
   ["5", "20", "90-cytokine response matrix + split-half reliability, well-level null", "no"],
   ["6", "21", "Receptor layer: binding matrix B, signs, competence R, D_binding", "no"],
   ["7", "22 (A+B)", "Representation screen; agonist/antagonist panel (Part B, new)", "no"],
   ["8", "30, 31", "Training set of all 90 cytokines; analytic baselines for every held-out ligand", "no"],
   ["9", "32, 40, 41, tests", "Stage B: arm flags, zero-shot run notebook, pilot summary", "<b>not run yet</b>"],
   ["10", "—", "Cross-cutting caveats and open issues", "—"]], [1.1*cm, 2.4*cm, 10.4*cm, 3.4*cm])

H2("Executive summary")
B(["<b>Data.</b> Parse Biosciences PBMC cytokine screen via <font name='DVM'>theislab/cellflow-datasets</font>: 9,697,974 cells × 2,000 HVGs, 12 donors, 90 cytokines + PBS, 18 cell types (13.1 GB; CC BY-NC 4.0).",
   "<b>06 — the only completed baseline.</b> With IL-15 seen in one donor (Donor8) and predicted for three others, CellFlow beats a 'control-mean' baseline in 3/3 donors: energy distance 16.5 vs 101.6, R² 0.958 vs 0.821. Training took 3.84 h and peaked at 3.07 GB GPU memory.",
   "<b>08 — ESM2 content matters, but a unique ID goes a long way.</b> Arm A (ESM2) e-dist 16.5; B (family one-hot, IL-15 identical to seven γ-chain relatives) 66.8; C (ESM2 shuffled across cytokines, still a unique ID for IL-15) 37.9; control 101.6. The notebook's own conclusion cell was never filled in.",
   "<b>09 — receptor mRNA is necessary, not sufficient.</b> 7 receptor-poor ligands give no response above the well-level null; TNF-α and IGF-1 are null despite abundant receptor. The 06 model predicts near-zero for <i>every</i> unseen ligand, so it gets the poor ones right for the wrong reason and misses IFN-γ (predicted/observed = 0.23).",
   "<b>20 — half the cytokines do not visibly respond.</b> 46 of 90 pass split-half r ≥ 0.5; the other 44 sit at 0.84–1.13× the well-level null. The naive random-PBS-split null would have called 35 of those 44 responders. <b>Any ligand-vs-PBS comparison needs the well-level null</b> (each cytokine × donor sits in one barcoding well).",
   "<b>21 — receptor layer.</b> 85/90 cytokines have annotated receptors; 4 antagonists carry sign −1; 5 have no annotation.",
   "<b>22A — no representation predicts response geometry well.</b> ESM2 ρ = 0.046 (reliable set, Mantel p = 0.18), indistinguishable from random vectors (0.035). Signed binding ρ = 0.108 (p = 0.0009) and family one-hot ρ = 0.134 (p = 0.0003) are weak but non-zero; binding − ESM2 is +0.060 with a bootstrap CI [−0.09, 0.21] (not demonstrated).",
   "<b>22B (new) — agonist/antagonist panel.</b> Binding puts both antagonist pairs at the maximum distance (2.0, percentile 100) as designed. ESM2 does <i>not</i> rank IL-1Ra near IL-1β (rank 32 and 4; percentile 25 raw, 34 centred), so the strong prediction is only partly met. IL-36α/IL-36Ra are near in ESM2 (rank 2 and 2) but their response distance cannot be interpreted (split-half r 0.42 and 0.15).",
   "<b>30–31 — baselines.</b> The 90-cytokine training set has 3,411,214 cells. The analytic 'cytokine-mean' baseline barely improves on 'control-mean' (mean e-dist 31.3 vs 34.0; wins in 57.4% of cases) — the bar a trained model must clear is low.",
   "<b>Stage B/C.</b> Arm flags (<font name='DVM'>--pert_rep</font>, <font name='DVM'>--gate</font>), a receptor gate, unit tests, a papermill run notebook and a pilot array are written. <b>None has run; no pilot result and no PILOT.md exist.</b> The 72-run main experiment has not been started."])

# ------------------------------------------------------------------ Part 0
H1("Part 0 — Notebooks 01–05: infrastructure and data")
H2("Source data (04, DATA.md)")
B(["File <font name='DVM'>pbmc_parse.h5ad</font>, 13,101,592,422 bytes, sha256 c72671b3…d4c9 (HF LFS), revision 4b235c9a…; opened backed (X never loaded).",
   "9,697,974 cells × 2,000 genes (all highly variable), CSR float32, 964.5 M non-zeros. X is <b>raw UMI integers</b> (max 1897, 100% integer, <font name='DVM'>uns['log1p']</font> empty) → every notebook normalises to CP10k then log1p. Because X holds only the 2,000 HVGs, CP10k is relative to HVG counts, not total transcripts (stated convention, kept across 07, 20, 30).",
   "12 donors, 91 conditions (90 cytokines + PBS), 18 cell types, 15 cytokine families. <font name='DVM'>uns</font>: <font name='DVM'>esm2_embeddings</font> (93 entries × 2,560-d, norms 9.5–14.8, <b>not L2-normalised</b>, raw cosine distance median ≈ 0.05 → nearly collinear), <font name='DVM'>donor_embeddings</font> (12 × 2,000, the donor's control mean over HVGs).",
   "obs columns include <font name='DVM'>bc1_well</font>: each cytokine × donor sits in <b>one</b> barcoding well, PBS spans 6 wells per donor — the origin of the well-confound found in 09/20."])
H2("01 — GPU inventory · 02 — environment · 03 — GPU verification")
B(["<b>01</b>: <font name='DVM'>sinfo</font> via <font name='DVM'>ssh max-login</font> on the MDC cluster: T4 (max001–010), A40 ×3 per node (max020–029), V100 and B300 nodes. No model saved, nothing trained.",
   "<b>02</b>: conda env <font name='DVM'>cellflow</font> (Python 3.11) with <font name='DVM'>cellflow-tools 0.0.9</font>, <font name='DVM'>jax[cuda12]</font>; <b>no rapids-singlecell</b>; env vars <font name='DVM'>XLA_PYTHON_CLIENT_PREALLOCATE=false</font>, HF/pip caches redirected off $HOME.",
   "<b>03</b>: on an A40 (46,068 MiB, driver 610.57, cc 8.6) JAX sees a CudaDevice; 4096² matmul works; jax 0.10.2. <font name='DVM'>cellflow.__version__</font> does not exist in 0.0.9 (expected AttributeError, recorded)."])
P("<b>Environment change.</b> Notebooks 01–09 ran on the MDC <i>max</i> cluster; from 2026-11-01 the project runs on the Cyfronet/TASK cluster (<font name='DVM'>SETUP.md</font>, <font name='DVM'>HARDWARE.md</font>): A100-80GB nodes (1 GPU, 72 GB host RAM, 42 CPUs), H100 ×4 nodes, A100 (akiv) nodes down. The new env has jax 0.10.2 + CUDA 12.9 wheels; <b>CUDA on the new cluster was not verified</b> when SETUP.md was written, and per-user QOS limits could not be read (accounting DB unreachable).", "small")
H2("05 — The 20-cytokine subset (<font name='DVM'>pbmc_subset.h5ad</font>)")
P("Deterministic recipe. <b>Donors (4):</b> Donor8 + the 3 donors with most PBS cells (Donor1 90,452; Donor12 71,179; Donor2 70,520). <b>Cytokines (20):</b> IL-15 + its 7 family members (IL-13, IL-2, IL-21, IL-4, IL-7, IL-9, TSLP) + 12 drawn with <font name='DVM'>default_rng(0)</font> (ADSF, BAFF, C5a, FGF-beta, GM-CSF, IFN-epsilon, IL-17B, IL-20, IL-31, IL-6, LT-alpha2-beta1, SCF) + PBS. All 18 cell types kept. Result: <b>1,114,081 cells × 2,000 genes</b> (0.33 GB, sha256 61883f33…), cells per donor Donor8 187,549 · Donor1 362,067 · Donor12 309,565 · Donor2 254,900; 275,215 PBS cells.")

# ------------------------------------------------------------------ 06
H1("Part 1 — Notebook 06: CellFlow baseline (IL-15 held out in 3 of 4 donors)")
H2("Question and design")
P("Reproduce CellFlow's PBMC tutorial (<font name='DVM'>100_pbmc</font>) on the 20-cytokine subset and test whether it predicts a cytokine's effect in donors where that cytokine was not observed. <b>Split:</b> IL-15 is in training for Donor8 only; IL-15 for Donor1, Donor2, Donor12 is the test set (so this is <b>donor-transfer</b>, not zero-shot ligand). PBS of all donors is available as source cells. Deviations from the tutorial: PCA on CPU (no rapids), no UMAPs, local subset file.")
H2("Data and preprocessing")
B(["Subset from notebook 05; <font name='DVM'>normalize_total(1e4)</font> + <font name='DVM'>log1p</font>; PCA with <font name='DVM'>centered_pca(n_comps=100)</font> fitted on training data, test projected with <font name='DVM'>project_pca</font>.",
   "Embedding sanity checks: 93 ESM2 vectors of 2,560-d, not L2-normalised; all 20 treated cytokines present (PBS needs none, it is the source distribution).",
   "Validation subsampling: 1,000 cells per training condition, 2,000 per test condition."])
H2("What was trained")
T([["Item", "Setting"],
   ["Model", "CellFlow, solver <font name='DVM'>otfm</font> (OT flow matching), deterministic condition encoder, <font name='DVM'>pooling='attention_token'</font>, 256-d condition embedding, output dropout 0.9"],
   ["Condition inputs", "perturbation covariate <font name='DVM'>cytokine_treatment</font> = ESM2 (2,560-d) through MLP [1024,1024] (dropout 0.5); sample covariate <font name='DVM'>donor</font> = donor embedding through MLP [256,256]; after pooling MLP [1024,1024]; <font name='DVM'>split_covariates=['donor']</font>"],
   ["Vector field", "time encoder [1024]×3, x-encoder [2048]×3, decoder [4096]×3, silu; concatenation conditioning; probability path constant_noise 0.5; matching <font name='DVM'>match_linear(ε=0.5, τ_a=τ_b=1)</font>"],
   ["Optimiser", "Adam 5e-5 inside <font name='DVM'>optax.MultiSteps(…, 20)</font> (20 micro-steps per update, so 500k iterations = 25k parameter updates)"],
   ["Schedule", "500,000 iterations, batch 1,024, validation every 20,000; seed 0 (CellFlow default); training cells ≈ the subset minus IL-15 in Donor1/2/12"],
   ["Hardware / cost", "A40 GPU (MDC). <b>Wall time 13,832 s = 3.84 h; peak GPU memory 3.07 GB</b> (JAX allocator, preallocation off)"],
   ["Saved model", "<font name='DVM'>models/06_baseline_CellFlow.pkl</font> — the only trained model object in the repository"]], [3*cm, 14.3*cm])
H2("Evaluation")
P("Predictions for IL-15 in each test (donor, cell type) are compared with the true IL-15 cells and with the <b>control-mean baseline</b> (the donor's own PBS cells, i.e. 'no effect'). Metrics: energy distance (<font name='DVM'>compute_e_distance</font>) in a 30-PC space fitted on a 100k-cell subsample of the whole dataset, and R² (<font name='DVM'>compute_r_squared</font>) in gene space after PCA reconstruction; groups need ≥ 20 cells. Acceptance rule: CellFlow better than control on both metrics.")
H2("Results")
t = d06.pivot_table(index="donor", columns="model", values=["e_distance", "r_squared"]); rows = [["donor", "e-dist CellFlow", "e-dist control", "R² CellFlow", "R² control"]]
for d, r in t.iterrows(): rows.append([d, f"{r[('e_distance','CellFlow')]:.2f}", f"{r[('e_distance','control_mean')]:.2f}", f"{r[('r_squared','CellFlow')]:.3f}", f"{r[('r_squared','control_mean')]:.3f}"])
mm = d06.groupby("model")[["e_distance", "r_squared"]].mean(); rows.append(["<b>mean</b>", f"<b>{mm.loc['CellFlow','e_distance']:.2f}</b>", f"<b>{mm.loc['control_mean','e_distance']:.2f}</b>", f"<b>{mm.loc['CellFlow','r_squared']:.3f}</b>", f"<b>{mm.loc['control_mean','r_squared']:.3f}</b>"])
T(rows, [3*cm, 3.4*cm, 3.4*cm, 3.4*cm, 3.4*cm]); IMG(f"{FIG}/f06.png", 15*cm, "Figure 06: per-donor energy distance and R² for CellFlow vs control-mean (derived from models/06_baseline_metrics_per_donor.csv).")
ct = c06.groupby(["cell_type", "model"])[["e_distance", "r_squared"]].mean().unstack(); rows = [["cell type", "e-dist CF", "e-dist ctrl", "R² CF", "R² ctrl"]]
for c, r in ct.iterrows(): rows.append([c, f"{r[('e_distance','CellFlow')]:.1f}", f"{r[('e_distance','control_mean')]:.1f}", f"{r[('r_squared','CellFlow')]:.2f}", f"{r[('r_squared','control_mean')]:.2f}"])
H3("By cell type (mean over the 3 test donors)"); T(rows, [4.2*cm, 3*cm, 3*cm, 3*cm, 3*cm])
B(["<b>Acceptance: PASS</b> — CellFlow better in 3/3 donors on both metrics. The effect is large for energy distance (6×) and moderate for R² (the control-mean baseline already has R² 0.82 in gene space, so R² has less room to separate models).",
   "Worst-predicted cell types by energy distance: CD8 Memory (77.4), CD16 Mono (67.3), MAIT (53.4). Best: B cells (18.1) and CD4 Naive (17.5).",
   "<b>Not tested here:</b> a truly unseen ligand (IL-15 was in training for Donor8); other ligands; seeds (single run, no variance estimate); any lower iteration count."])
P("Statistics: none beyond descriptive means over 3 donors; there is no confidence interval, test or seed replicate.", "small")

# ------------------------------------------------------------------ 07
H1("Part 2 — Notebook 07: what does the learned condition embedding encode?")
H2("Question")
P("Has CellFlow's condition encoder organised cytokines by <b>receptor family</b>, and if so did it learn that or inherit it from ESM2? Part I interrogates the 06 model; Part II asks whether the <i>data</i> (responses) have the same structure.")
H2("Part I — embeddings from the 06 model (no training)")
B(["<font name='DVM'>cf.get_condition_embedding</font> on 93 cytokines × 4 donors (372 conditions, 256-d; variance is 0 since the encoder is deterministic). Spaces: ESM2 raw, ESM2 centred (mean over 93 removed), learned per donor and donor-mean.",
   "Statistics: cosine silhouette with <font name='DVM'>cytokine_family</font> labels vs <b>100 label permutations</b> (p floor 0.01); within- vs between-group cosine distance for 4 shared-receptor groups. Families with one trained member are dropped from silhouette.",
   "<b>Caveat from the notebook:</b> only IL-6 and IFN-epsilon of the 13 cytokines in the receptor groups were trained; the rest pass through the trained encoder zero-shot, so Part I mostly tests what the encoder does to ESM2.",
   "<b>Result (notes.md):</b> the learned embedding has not rediscovered receptor structure; where receptor groups are tight in it, the structure is inherited from ESM2."])
H2("Part II — do the responses have family structure? (full dataset)")
P("<b>Response vector</b> per cytokine: for each donor × cell type, pseudobulk mean log1p(CP10k) minus the same donor's PBS mean (≥ 20 cells both sides), averaged over donors and concatenated over the 14 cell types with a delta for every cytokine (HSPC, ILC, Plasmablast, pDC dropped). Reliability: split-half correlation donors 1–6 vs 7–12. Cached group means: <font name='DVM'>pbmc_parse_pseudobulk_log1p_means.h5ad</font>.")
T([["space", "14 trained cytokines (4 families) silhouette, p", "77 cytokines (10 families)", "40 reliable responders"],
   ["response (data)", "−0.021, p = 0.10", "−0.118 (null −0.18), p = 0.01", "−0.058 (null −0.21), p = 0.01"],
   ["ESM2 raw", "0.097, p = 0.04", "−0.142, p = 0.01", "−0.211, p = 0.04"],
   ["ESM2 centred", "0.139, p = 0.02", "−0.020, p = 0.01", "−0.080, p = 0.01"],
   ["learned (donor mean)", "−0.117, p = 0.17", "−0.191, p = 0.01", "−0.120, p = 0.01"]], [3.6*cm, 5*cm, 4.4*cm, 4.3*cm])
P("Source: notes.md §1. A negative silhouette that still beats the permutation null (p = 0.01) means the label structure is <i>less negative than random</i>, not that clusters exist. Only the <b>interferons</b> cohere in the data (per-family silhouette +0.24); every other family is negative. ESM2 groups sequence families (complement, common β chain, IL-17); the learned space groups complement, IL-17 and IL-6/IL-12.", "small")
IMG("models/07_response_vs_embedding_structure.png", 17*cm, "Figure 07: left — family silhouette (observed, purple) against 100 permuted-label draws, 77 cytokines / 10 families. Right — percentile of within-group distance vs random groups of the same size (low = tight; * ≤ 5th).")
T([["receptor group", "response (data)", "ESM2 raw", "ESM2 centred", "learned"],
   ["IFNAR (IFN-α1, -β, -ε, -ω)", "0.1*", "0.2*", "0.0*", "14.4"], ["CSF1R (M-CSF, IL-34)", "21.8", "9.8", "3.6*", "26.7"],
   ["gp130 (IL-6, -11, LIF, OSM, CT-1)", "84.2", "0.4*", "0.1*", "1.7*"], ["IL17RA/RC (IL-17A, -17F)", "23.1", "0.2*", "0.3*", "0.1*"]], [5.5*cm, 3*cm, 3*cm, 3*cm, 3*cm])
BOX("<b>Answer (notes.md).</b> PBMC responses cluster by family only weakly; the single shared-receptor group tight in the data is IFNAR. gp130 and IL-17A/F are tight in ESM2 and the model but their cells do not respond alike (the ligand-specific α chains decide a blood-cell response; IL-11 and LIF barely respond). 'Embeddings group by sequence family; cells group by the receptors they actually carry; the two agree only for IFNAR.'")

# ------------------------------------------------------------------ 08
H1("Part 3 — Notebook 08: does ESM2 do real work? (family one-hot and shuffled-ESM2 controls)")
H2("Design")
P("Identical pipeline, split, seed (0) and 500k-iteration training to notebook 06. <b>Only the cytokine representation changes</b>, and so does the first encoder layer's input size for arm B.")
T([["arm", "cytokine representation", "what it tests"],
   ["A", "ESM2 (2,560-d)", "the 06 baseline; <b>reused from the saved 06 model</b> because the training PCA matched (reproducibility checked against 06's CSV: R² identical to 4 decimals, e-distance within 0.03 per the notes)"],
   ["B", "family one-hot over <font name='DVM'>cytokine_family</font> (10 families); IL-15 is identical to IL-2/4/7/9/13/21/TSLP", "is coarse receptor-family information enough?"],
   ["C", "ESM2 vectors permuted across the 20 cytokine names (none keeps its own)", "control: IL-15's vector is still a unique, consistent ID"]], [1.3*cm, 8.3*cm, 7.7*cm])
H2("What was trained")
P("Arms B and C: two new CellFlow models, each 500k iterations, batch 1,024, same architecture and optimiser as 06 (sbatch job 3375645 per notes.md; GPU type not recorded in the files I read). Arm A: the 06 model, not retrained. <b>Per-arm wall time and VRAM were not extracted for this report.</b>")
H2("Results (cell_type = 'all', held-out IL-15)")
a8 = m08[m08.cell_type == "all"]; t = a8.pivot_table(index="donor", columns="model", values="e_distance"); rows = [["donor", "A ESM2", "B family", "C shuffled", "control-mean"]]
for d, r in t.iterrows(): rows.append([d] + [f"{r[k]:.1f}" for k in ["A_esm2", "B_family_onehot", "C_esm2_shuffled", "control_mean"]])
mm = a8.groupby("model")[["e_distance", "r_squared"]].mean(); rows.append(["<b>mean e-dist</b>"] + [f"<b>{mm.loc[k,'e_distance']:.1f}</b>" for k in ["A_esm2", "B_family_onehot", "C_esm2_shuffled", "control_mean"]])
rows.append(["<b>mean R²</b>"] + [f"<b>{mm.loc[k,'r_squared']:.3f}</b>" for k in ["A_esm2", "B_family_onehot", "C_esm2_shuffled", "control_mean"]])
ctl, A_ = mm.loc["control_mean", "e_distance"], mm.loc["A_esm2", "e_distance"]
rows.append(["gap closed vs A, e-dist"] + ["1.00", f"{(ctl-mm.loc['B_family_onehot','e_distance'])/(ctl-A_):.2f}", f"{(ctl-mm.loc['C_esm2_shuffled','e_distance'])/(ctl-A_):.2f}", "0.00"])
T(rows, [4.2*cm, 3*cm, 3*cm, 3*cm, 3.3*cm]); IMG(f"{FIG}/f08.png", 15.5*cm, "Figure 08: per-donor energy distance and R² for the three arms and control-mean (derived from models/08_family_vs_esm2_metrics.csv).")
B(["<b>Ordering on both metrics:</b> A (ESM2) &gt; C (shuffled ESM2) &gt; B (family one-hot) &gt; control-mean. Relative to the gap between control-mean and A, the shuffled arm closes ≈ 75% and the family one-hot ≈ 41% (e-distance, means over 3 donors).",
   "<b>Donor heterogeneity:</b> C ≈ A in Donor1 (10.6 vs 7.3) and Donor12 (25.8 vs 22.2) but much worse in Donor2 (77.4 vs 20.1, near control 82.7). With 3 donors and one seed the 'mean' is carried by that spread.",
   "<b>How to read it (notebook caveat):</b> IL-15 is in training for Donor8, so C's shuffled vector works as an ID; C ≈ A therefore says ESM2 <i>content</i> is not needed for <i>donor transfer</i>, not that the representation is ignored. B shows a generic γ-chain response is not enough. Neither speaks to unseen cytokines — that needs a split where IL-15 is absent for all donors (Stage B/C)."])
BOX("<b>Status.</b> The notebook's own 'Conclusion' cell still reads '(filled in after the run)'. The interpretation above is mine, from the saved CSVs, not the notebook's. No seeds or confidence intervals exist for this experiment.")

# ------------------------------------------------------------------ 09
H1("Part 4 — Notebook 09: receptor competence vs observed response vs the 06 model")
H2("Question")
P("Do ligands whose receptors are barely expressed on PBMCs fail to produce a response, and does the 06 model know that?")
H2("Data decisions")
B(["<b>Responses:</b> full HF dataset (13 ligands + PBS over 12 donors in this notebook's scope; 8 of the 13 are absent from the 20-cytokine subset).",
   "<b>Receptor expression:</b> the HF matrix contains only 4 of 19 receptor genes, so Parse's <b>PBS pseudobulks</b> (raw counts, ~40k genes) were used: 24 files pulled out of the remote 44 GB archive by HTTP range requests (330 MB transferred) — helper <font name='DVM'>scripts/http_range_file.py</font>. GFRA4 is not annotated, so PSPN is scored on RET alone.",
   "<b>Cell-type mapping</b> from Parse labels to HF labels by name; PBS counts agree for monocytes/DC/B/NK/HSPC (r ≥ 0.9), less for CD4 naive/memory, Treg, MAIT (0.71–0.98), poorly for ILC (0.14)."])
H2("Method")
B(["Per donor × ligand × cell type, ligand and PBS are subsampled to the same n = min(n_lig, n_PBS, 500), n ≥ 30. DE: Wilcoxon, BH FDR &lt; 0.05, |log2FC| &gt; 0.5. Energy distance in 50 PCs fitted on a balanced subsample. Median over donors.",
   "Receptor score: mean log1p(CPM) over listed subunits, averaged over donors (sensitivity: scarcest subunit)."])
H2("Two method issues found (important for everything after)")
T([["null", "energy distance", "DE genes"], ["random PBS halves", "1.74", "0.0"], ["one PBS well vs the other PBS wells (same donor)", "9.17", "1.75"]], [8*cm, 3.5*cm, 3.5*cm])
B(["<b>Well confound.</b> Each cytokine × donor is in one barcoding well; PBS spans 6. A random-halves null is too low: against it every receptor-poor ligand looks like a weak responder in 14–16 cell types; against the well null none does.",
   "<b>Energy-distance bias.</b> <font name='DVM'>compute_e_distance</font> includes the zero self-distances (a V-statistic), biased upward by ≈ (σ_X + σ_Y)/n. Floors at different n are not comparable → every comparison within a donor × cell type must use the same n."])
H2("Observed responses (well floor)")
T([["ligand", "group", "median DE", "median e-dist", "cell types &gt; 2× well floor (e-dist)", "cell types &gt; floor (DE)"],
   ["EGF", "poor", "2.0", "9.5", "0/16", "0"], ["FGF-beta", "poor", "2.0", "10.5", "1/16", "0"], ["HGF", "poor", "0.5", "7.2", "0/15", "0"], ["IGF-1", "poor", "3.5", "9.9", "1/15", "1"],
   ["VEGF", "poor", "2.0", "7.0", "0/15", "0"], ["GDNF", "poor", "1.0", "6.2", "0/16", "0"], ["PSPN", "poor", "1.0", "7.4", "0/15", "0"],
   ["GM-CSF", "rich", "8.0", "21.8", "9/17", "7"], ["IFN-γ", "rich", "35.0", "40.8", "16/16", "13"], ["IL-15", "rich", "150.3", "165.9", "16/16", "16"],
   ["IL-2", "rich", "94.0", "103.1", "16/16", "15"], ["IL-4", "rich", "46.3", "55.0", "14/16", "14"], ["TNF-α", "rich", "2.0", "6.8", "2/17", "1"]], [2.4*cm, 1.6*cm, 2.2*cm, 2.6*cm, 5*cm, 3.5*cm])
B(["TNF-α is effectively null despite high TNFRSF1A/1B; IGF-1 is null despite abundant IGF1R in T cells → <b>receptor mRNA is necessary but not sufficient</b>."])
H2("Receptor expression vs response (206 ligand × cell-type pairs)")
T([["pairs", "Spearman ρ (DE)", "Spearman ρ (e-dist)"], ["all", "0.55 (p &lt; 1e-15)", "0.53"], ["receptor-rich only", "0.34 (p = 6e-4)", "0.22 (p = 0.03)"], ["receptor-poor only", "0.22 (p = 0.02)", "0.24 (p = 0.01)"]], [5*cm, 5*cm, 5*cm])
IMG(f"{FIG}/f09.png", 10.5*cm, "Figure 09: receptor score vs number of DE genes, coloured by group (derived from models/09_receptor_vs_response.csv). The pooled correlation is largely the two-group split.")
H2("The 06 model's predictions for these ligands (model's 100-PC space, 4 donors, matched n; well floor 21.3, random floor 13.1)")
T([["ligand", "status in model", "observed / floor", "predicted / floor", "pred / obs"],
   ["EGF", "unseen", "1.10", "0.77", "0.71"], ["FGF-beta", "trained", "1.11", "0.79", "0.71"], ["HGF", "unseen", "0.92", "0.73", "0.79"], ["IGF-1", "unseen", "1.35", "0.86", "0.63"],
   ["VEGF", "unseen", "0.90", "0.78", "0.87"], ["GDNF", "unseen", "1.01", "0.90", "0.90"], ["PSPN", "unseen", "0.99", "0.88", "0.89"], ["GM-CSF", "trained", "2.49", "2.07", "0.83"],
   ["IL-15", "trained (Donor8)", "9.71", "5.85", "0.60"], ["TNF-α", "unseen", "0.97", "1.38", "1.43"], ["<b>IFN-γ</b>", "<b>unseen</b>", "<b>4.03</b>", "<b>0.91</b>", "<b>0.23</b>"], ["IL-4", "trained", "5.06", "6.61", "1.31"], ["IL-2", "trained", "5.92", "4.84", "0.82"]], [2.6*cm, 3.6*cm, 3.6*cm, 3.6*cm, 3.2*cm])
BOX("<b>Answer.</b> The model does not predict a substantial effect for receptor-poor ligands — matching the observed null — but because it predicts a near-floor response for <i>every</i> unseen ligand, including IFN-γ (observed 4× floor). From an ESM2 vector alone it cannot separate 'receptor present' from 'receptor absent'. Receptor competence dominates whether PBMCs respond at all, and the model has no access to it. This motivates the receptor gate in Stage B.")

# ------------------------------------------------------------------ 20
H1("Part 5 — Notebook 20: 90-cytokine response matrix and split-half reliability")
H2("Method")
B(["<b>Source:</b> full dataset (9.7 M cells), CPU only. <font name='DVM'>scripts/20_pseudobulk.py</font> streams X once (39 s) and caches sums per (donor, cytokine, cell type, <font name='DVM'>bc1_well</font>) plus a seeded random 50/50 PBS split → 20,524 groups, 90 cytokines, 18 cell types.",
   "<b>Delta</b> (donor, cytokine, cell type) = mean over cytokine cells − mean over that donor's PBS cells of the same cell type; both sides ≥ 20 cells. 15,204 deltas of 19,440 possible. Averaged over donors that pass.",
   "<b>Response vector</b> = concatenation over cell types that have a delta for <i>every</i> cytokine: 14 cell types (HSPC, ILC, Plasmablast, pDC dropped; ILC missing for 47 cytokines, Plasmablast for 75) × 2,000 genes = <b>28,000</b>-d; matrix 90 × 28,000.",
   "<b>Reliability:</b> Pearson r between the response vectors computed from Donors 1–6 and Donors 7–12. <b>Null:</b> well-level (one PBS well vs the other PBS wells of the same donor) vs the naive random PBS split."])
H2("Results")
T([["category", "count"], ["reliable (r ≥ 0.5)", "46"], ["low r, response ≤ 2× well null ('no detectable response')", "44"]], [10*cm, 3*cm])
B(["All 44 low-r cytokines lie at 0.84–1.13× the well-level null (none &gt; 1.5×) → 'no detectable response', not 'donors disagree'. Under the naive random null, 35 of the 44 would have exceeded 2× and been called responders.",
   "Top reliable: IFN-β 0.947, IL-15 0.923, IL-7 0.921, IL-4 0.920, IL-2 0.916, CD40L 0.915, IFN-ω 0.913, IFN-γ 0.897, IL-1β 0.878, GM-CSF 0.873. The brief expected ~40 reliable responders; this run gives 46 at r ≥ 0.5.",
   "<b>Caveat in the notebook:</b> r ≥ 0.5 does not prove a response — 32 of the 46 reliable cytokines (e.g. EGF, Decorin, Leptin, IGF-1 with r ≈ 0.5) are also &lt; 1.5× the well null, so the whole-vector L2 test is blunt, and a shared well/plate-layout effect could contribute to reliability."])
IMG("models/20_reliability_vs_magnitude.png", 16*cm, "Figure 20: split-half r vs response magnitude (median ‖Δ‖) normalised to the well-level null (left) or the naive random-split null (right); purple = reliable, teal = not. Dashed: r = 0.5 and 2× null.")
H3("Reliable cytokines (all 46, split-half r)")
rr = rel[rel.reliable].sort_values("r_split_half", ascending=False); rows = [["cytokine", "family", "r"]] + [[r.cytokine, r.family, f"{r.r_split_half:.3f}"] for r in rr.itertuples()]
half = (len(rows) - 1 + 1) // 2 + 1; L, R_ = rows[:half], [rows[0]] + rows[half:]
while len(R_) < len(L): R_.append(["", "", ""])
T([a + b for a, b in zip(L, R_)], [2.6*cm, 4.4*cm, 1.3*cm, 2.6*cm, 4.4*cm, 1.3*cm])

# ------------------------------------------------------------------ 21
H1("Part 6 — Notebook 21: the receptor layer (binding, sign, competence)")
H2("Sources and construction")
B(["<b>Binding matrix B (90 ligands × subunits):</b> OmniPath <font name='DVM'>LigRecExtra</font> (v1.0.12, academic licence). Complexes from the curated trio <b>CellPhoneDB, CellChatDB, ICELLNET</b>; origin resource kept per edge. Evidence A = ≥ 2 of the trio (83 ligands), B = 1 of the trio (Decorin), C = other OmniPath only (Noggin), none (5).",
   "<b>Sign:</b> CellChatDB-cofactors inhibition edges. Antagonist = −1 on its own receptor subunits and the receptors of the ligands it neutralises: <b>IL-1Ra, IL-36Ra, Noggin, Decorin</b>. (NicheNet is not used anywhere.)",
   "<b>Receptor abundance R:</b> Parse PBS pseudobulks, 18 archive members fetched by range requests (117 MB transferred). Per (donor, cell type) log1p(CPM).",
   "<b>Competence</b> per (ligand, donor, cell type): each complex is an alternative receptor; all subunits required → <font name='DVM'>competence_min</font> = max over complexes of the <i>minimum</i> subunit abundance (mean version = sensitivity). The mean badly overstates IL-2 competence in B naive (min 0.90 vs mean 2.86).",
   "<b>D_binding</b>: 90×90 distance from B, signed and unsigned. Decoys excluded (IL1R2, IL13RA2, TNFRSF11B, TNFRSF6B, TNFRSF10C/D, IL22RA2) — a curation choice flagged in <font name='DVM'>binding_edges.parquet</font>."])
H2("Coverage and gaps")
B(["<b>85/90 annotated.</b> No annotation: ADSF (RETN), IL-17D, IL-32-beta, LT-alpha1-beta2, LT-alpha2-beta1 → all-zero B rows; D_binding fixes their distance to everything to 1.0 (flagged).",
   "Subunits per cytokine: median 2, max 16 (Noggin, inherited from its BMP/GDF targets). IFN-alpha1 annotated through alias IFNA13. GFRA4 absent from Parse annotation → <b>PSPN competence NaN</b>. Plasmablast has 11 donors.",
   "Assertion test passed: IL-1Ra and IL-36Ra are negative; IL-1Ra vs IL-1β signed distance 2.0 vs unsigned 0.0."])
IMG("models/21_coverage.png", 14*cm, "Figure 21a: evidence level (left) and number of receptor subunits per cytokine (right).")
IMG("models/21_D_binding.png", 15*cm, "Figure 21b: signed (left) and unsigned (right) D_binding. Signed distances make antagonist–agonist pairs maximal; most of the 90×90 matrix is the constant 1.0 background.")
P("Files: <font name='DVM'>data/binding_B.parquet, binding_B_long.parquet, binding_edges.parquet, competence_R.parquet (18,275 rows), receptor_abundance_R.parquet (215 × 123), D_binding_signed/unsigned.{npy,csv}, receptor_coverage.csv</font>. Licence of Parse expression data: CC BY-NC 4.0.", "small")

# ------------------------------------------------------------------ 22
H1("Part 7 — Notebook 22: representation screen (Part A) and agonist/antagonist panel (Part B)")
H2("Part A — does any condition representation predict response geometry?")
P("<b>Design.</b> Each candidate representation → 90×90 cosine-distance matrix, compared with D_response (cosine distance between response vectors, 14 cell types × 2,000 genes). <b>Candidates:</b> ESM2 raw; ESM2 centred; signed binding (nb 21); family one-hot; UniProt Function text (all-MiniLM-L6-v2 sentence embeddings, mean over ≤ 200-word chunks); <b>constant</b> (control: all distances 1 → ρ undefined); <b>random Gaussian</b> 64-d (seed 0; calibration). <b>Subsets:</b> the 46 reliable responders and all 90, plus per cell type.")
P("<b>Statistics.</b> (i) Spearman ρ between the flattened upper triangles. (ii) <b>Mantel permutation test</b>: candidate-matrix labels permuted jointly on rows and columns, 10,000 draws, seed 0, one-sided 'greater', p = (1+#≥obs)/(1+10,000). (iii) Distance correlation from the two distance matrices with its own label-permutation test (the V-statistic dCor is biased upward — the constant control gets 0.66 — so only its permutation p is used). <b>Uncorrected</b> p-values, 7 representations × 2 subsets.")
rows = [["representation", "subset", "ρ", "Mantel p", "dCor", "dCor p", "n"]]
for r in rs.itertuples(): rows.append([r.representation, r.subset, "NaN" if np.isnan(r.rho) else f"{r.rho:.4f}", f"{r.mantel_p:.4f}", f"{r.dcor:.3f}", f"{r.dcor_p:.4f}", r.n])
T(rows, [3.3*cm, 2.2*cm, 2*cm, 2.3*cm, 2*cm, 2.3*cm, 1.2*cm])
rows = [["contrast", "subset", "mean ρ difference", "95% bootstrap CI", "P(diff ≤ 0)"]] + [[r.contrast, r.subset, f"{r.mean_diff:+.3f}", f"[{r.ci_lo:.3f}, {r.ci_hi:.3f}]", f"{r.frac_boot_le_0:.3f}"] for r in bvs.itertuples()]
H3("Does binding beat ESM2? (object-level bootstrap of the ρ difference)"); T(rows, [4.6*cm, 2*cm, 3*cm, 4*cm, 2.8*cm])
B(["<b>ESM2 does not predict response geometry:</b> ρ = 0.046 raw (p = 0.18) and 0.044 centred (p = 0.10) on reliable cytokines; −0.017 / 0.003 on all 90 — indistinguishable from random vectors (0.035, p = 0.13).",
   "<b>Only signed binding and family one-hot show signal, and it is weak</b> (best ρ explains ≈ 2% of rank variance). Binding − ESM2 is +0.060 (CI [−0.09, 0.21]) → directionally positive, <b>not demonstrated</b>; binding does not beat family one-hot (−0.027), so its signal may be largely the family signal.",
   "Per cell type (reliable set): binding Mantel p &lt; 0.035 in all 14; family ρ highest in myeloid cells (CD14 Mono 0.143, cDC 0.149), lower in T cells (CD4 Memory 0.080, Naive 0.095). No cell-type correction.",
   "Caveats: 46 (not ~40) reliable responders; 5 cytokines have all-zero binding rows; IFN-ω has no UniProt Function text (embedded from its gene symbol)."])
IMG("models/22_response_vs_best_vs_esm2.png", 14*cm, "Figure 22a: D_response vs the best representation (family one-hot) and ESM2 raw, reliable (top) and all-90 (bottom). One-hot cosine distance is exactly 0 (same family) or 1 (different), hence two vertical lines.")

H2("Part B (new) — agonist / antagonist panel")
P("<b>Question.</b> Does a <i>structural</i> representation (ESM2) confuse an agonist with its competitive antagonist? <b>Pairs:</b> IL-1β / IL-1Ra (IL1R1) and IL-36α / IL-36Ra (IL1RL2) as antagonist pairs; IL-2 / IL-15 (IL2RB+IL2RG) and M-CSF / IL-34 (CSF1R) as positive controls. <b>Stated prediction:</b> antagonist pairs near in ESM2, far in signed binding, far in response; controls near in all three. For each pair and space: cosine distance, rank of the partner among the 89 others (from each side; ties averaged), and percentile of the pair distance among all 4,005 pairs (0 = closest).")
H3("Reliability check first (split-half r ≥ 0.5 required for BOTH members for the response distance to be interpretable)")
T([["pair", "r member A", "r member B", "both reliable?", "interpretation"],
   ["IL-1β / IL-1Ra", "0.878", "0.573", "yes", "response distance usable"], ["IL-36α / IL-36Ra", "0.419", "0.152", "<b>NO</b>", "<b>both 'no detectable response'; response column uninterpretable (reported, not dropped)</b>"],
   ["IL-2 / IL-15", "0.916", "0.923", "yes", "usable"], ["M-CSF / IL-34", "0.785", "0.655", "yes", "usable"]], [3.4*cm, 2.3*cm, 2.3*cm, 2.5*cm, 6.8*cm])
sp_ = ["esm2_raw", "esm2_centred", "binding", "family_onehot", "uniprot_text", "random_gauss", "RESPONSE (observed)"]
rows = [["pair", "space", "distance", "rank a→b", "rank b→a", "percentile"]]
for pr in ag.pair.unique():
    for s in sp_:
        r = ag[(ag.pair == pr) & (ag.space == s)].iloc[0]; rows.append([pr if s == "esm2_raw" else "", s + (" †" if (s.startswith("RESPONSE") and not r.response_interpretable) else ""), f"{r.distance:.3f}", f"{r.rank_b_from_a:.0f}", f"{r.rank_a_from_b:.0f}", f"{r.percentile:.0f}"])
T(rows, [3.6*cm, 4.2*cm, 2.4*cm, 2.3*cm, 2.3*cm, 2.3*cm]); P("† response distance NOT interpretable (a member has split-half r &lt; 0.5). 'constant' (all distances 1, percentile 50 by construction) omitted; it is in data/agonist_antagonist_panel.csv.", "small")
IMG("models/22_agonist_antagonist.png", 17*cm, "Figure 22b: percentile of pair distance in every space. Red = antagonist pairs, blue = positive controls; hollow marker = response distance not interpretable.")
B(["<b>Binding behaves as designed:</b> both antagonist pairs sit at distance 2.0 (rank 89, percentile 100); both positive controls are at the very bottom (IL-2/IL-15 0.333 rank 1; M-CSF/IL-34 0.000 rank 1).",
   "<b>IL-1β / IL-1Ra:</b> ESM2 does <i>not</i> rank IL-1Ra near IL-1β — rank 32 (IL-1β's list) and 4 (IL-1Ra's list); percentile 25 raw, 34 centred. Response distance is far (1.281, percentile 92, ranks 66/73). So the response differs but ESM2 only weakly says 'similar': the strong form of the prediction ('IL-1Ra is IL-1β's nearest neighbour in ESM2') is <b>not</b> met; the weak form (not far) is.",
   "<b>IL-36α / IL-36Ra:</b> ESM2 near (rank 2 and 2; percentile 7 raw, 2 centred), binding far — but the response comparison is unusable because both have no reproducible response. This pair cannot show ESM2 'failing on a functional distinction'.",
   "<b>Controls:</b> IL-2/IL-15 is near in every informative space (ESM2 percentile 1; response rank 1, percentile 0). M-CSF/IL-34 is near in ESM2 (percentile 10 / 4) and binding (0) and moderately near in response (0.682, rank 5/12, percentile 21).",
   "Family one-hot trivially puts all four pairs at 0 (percentile 5) because members share a family; random Gaussian and the constant are uninformative baselines. This is a 4-pair descriptive panel, with no statistical test."])

# ------------------------------------------------------------------ 30/31
H1("Part 8 — Notebooks 30 and 31: training set for all 90 cytokines and analytic baselines")
H2("30 — build <font name='DVM'>data/pbmc_train90.h5ad</font>")
B(["<b>Recipe:</b> 4 donors (Donor8, Donor1, Donor12, Donor2 — as in 05), <b>all 90 cytokines</b>, <b>cap 2,000 cells per (donor, cytokine, cell_type)</b> (random, seed 0), PBS in full. X = log1p(CP10k over the 2,000 HVGs); raw counts in <font name='DVM'>layers['counts']</font>.",
   "<b>PCA:</b> 100 PCs fitted on ≤ 200k random cells <b>excluding IL-2 and IFN-γ</b> (the pilot's held-out ligands), applied to all cells; explained variance of the 100 PCs = <b>0.497</b>. Stored in <font name='DVM'>obsm['X_pca']</font>, <font name='DVM'>uns['pca']</font>. Same PCA for every arm and baseline.",
   "Read + normalise took 17 s; file written with gzip (3.16 GB)."])
T([["quantity", "value"], ["cells total", "<b>3,411,214</b>"], ["PBS cells", "275,215"], ["ligand cells", "3,135,999"], ["cells per donor", "Donor1 948,373 · Donor12 946,135 · Donor2 884,495 · Donor8 632,211"],
   ["(donor, cytokine, cell type) groups", "6,480 possible · 6,436 non-empty · 5,278 with ≥ 20 cells · 449 at the 2,000 cap"], ["(donor, ligand) pairs present", "360 of 360"], ["cells per ligand", "min 20,856 (APRIL) · median 34,700 · max 44,469"]], [5*cm, 12.3*cm])
IMG(f"{FIG}/f30.png", 13.5*cm, "Figure 30: fraction of (donor, ligand) groups with ≥ 20 cells per cell type (derived from data/pbmc_train90_coverage.csv). HSPC 0.26, ILC 0.13, pDC 0.32, Plasmablast 0.04.")
BOX("<b>Note for the plan.</b> The Stage B brief assumed '~700k cells'. The built training set is <b>3.41 M cells</b> (4.9× larger), so the argument that 500k iterations is 'overkill for ~700k cells' must be re-examined against the real size. The pilot (not yet run) is what would decide it.")
H2("31 — analytic baselines for every (held-out ligand, donor, cell type)")
P("Leave-one-ligand-out over all 90 ligands, for each (donor, cell type) with ≥ 20 cells for the ligand and PBS. <b>control_mean</b> = the donor's PBS cells (no effect). <b>cytokine_mean</b> = those PBS cells translated by the mean delta of the <i>other 89</i> ligands in the same (donor, cell type) (held-out ligand never in its own mean), in PCA space for energy distance and gene space for R². Both sides subsampled to ≤ 1,000 cells (seed 0) so floors share n. CPU only (≈ 56 min); output <font name='DVM'>data/baselines_train90.csv</font> (10,554 rows, 5,277 cases). No well-level null is applied here.")
bb = base.groupby("baseline")[["e_distance", "r_squared"]].mean(); pv = base.pivot_table(index=["ligand", "donor", "cell_type"], columns="baseline", values=["e_distance", "r_squared"])
T([["baseline", "mean e-distance", "mean R²"], ["control_mean", f"{bb.loc['control_mean','e_distance']:.2f}", f"{bb.loc['control_mean','r_squared']:.4f}"], ["cytokine_mean", f"{bb.loc['cytokine_mean','e_distance']:.2f}", f"{bb.loc['cytokine_mean','r_squared']:.4f}"]], [5*cm, 4*cm, 4*cm])
P(f"cytokine_mean beats control_mean on e-distance in <b>{100*(pv[('e_distance','cytokine_mean')]<pv[('e_distance','control_mean')]).mean():.1f}%</b> of cases and on R² in <b>{100*(pv[('r_squared','cytokine_mean')]>pv[('r_squared','control_mean')]).mean():.1f}%</b> (n = {len(pv)}).")
rows = [["held-out ligand", "control_mean e-dist", "cytokine_mean e-dist", "control R²", "cytokine R²"]]
for l in ["IL-2", "IFN-gamma"]:
    t = base[base.ligand == l].groupby("baseline")[["e_distance", "r_squared"]].mean(); rows.append([l, f"{t.loc['control_mean','e_distance']:.1f}", f"{t.loc['cytokine_mean','e_distance']:.1f}", f"{t.loc['control_mean','r_squared']:.3f}", f"{t.loc['cytokine_mean','r_squared']:.3f}"])
T(rows, [3.5*cm, 3.6*cm, 3.9*cm, 3*cm, 3*cm]); IMG(f"{FIG}/f31.png", 14*cm, "Figure 31: mean energy distance of the two analytic baselines, by cell type, over all 90 held-out ligands (derived from data/baselines_train90.csv).")
P("Reading: the cytokine-mean baseline is only slightly better than doing nothing (a 2–4% e-distance gain for IL-2 and IFN-γ). For IL-2 and IFN-γ the distance to the truth is dominated by monocytes (CD14 Mono ≈ 220 and 406) — a model that does not move monocytes correctly cannot close the gap.", "small")

# ------------------------------------------------------------------ Stage B
H1("Part 9 — Stage B (flags, pilot, convergence): what exists and what has not run")
BOX("<b>Nothing in this part has produced a result.</b> The code is written but unverified: the unit tests (job 21691739) and the slim-file build (job 21691752) were still pending in the queue when this report was generated, so no training run, no pilot, no PILOT.md, and no Stage C run exist.")
H2("Implemented (in the repository, uncommitted)")
T([["component", "file", "what it does"],
   ["CellFlow fork", "cellflow_fork/cellflow/", "copy of cellflow-tools 0.0.9 edited in place (no git branches); loaded ahead of site-packages and asserted in the run notebook"],
   ["arm flags", "cellflow_fork/cellflow/arms.py", "<font name='DVM'>--pert_rep</font> {constant, esm2, binding, binding_esm2, family_onehot} and <font name='DVM'>--gate</font> {none, receptor, receptor_permuted, mlp_full_basal}; builds the pert reps and gate features and the <font name='DVM'>prepare_data</font>/<font name='DVM'>prepare_model</font> kwargs; <font name='DVM'>split_covariates=['donor','cell_type']</font>"],
   ["receptor gate", "networks/_velocity_field.py", "<font name='DVM'>CompetenceGate</font>: bias-free narrow MLP, output <font name='DVM'>tanh(|z|)</font> ∈ [0,1) multiplying the condition (ligand) latent — a mask, not a magnitude predictor; zero features ⇒ zero latent"],
   ["seed fix", "training/_trainer.py, model/_cellflow.py", "the trainer's JAX/NumPy RNGs were hard-coded to 0, so 'seeds' would have shared one batch stream; now seeded"],
   ["unit tests", "tests/test_arms.py", "zero competence ⇒ gated latent ≈ 0; permuting receptor features changes the output; constant pert_rep ⇒ identical ligands; tiny end-to-end train+predict for 7 arm combinations (CPU)"],
   ["slim data + basal means", "notebooks/32_slim_and_basal.ipynb", "expression-free copy of the training file (fast loading) and PBS mean expression per (donor, cell type) for the <font name='DVM'>mlp_full_basal</font> gate"],
   ["run notebook", "notebooks/40_run_zeroshot.ipynb", "papermill-parameterised: removes one ligand from ALL donors, trains, logs validation e-distance every 5k iterations with wall time, saves predictions, peak VRAM"],
   ["pilot array", "scripts/run_stageB_array.sbatch", "8 tasks = {IL-2, IFN-γ} × {constant/none, esm2/none, binding/none, binding/receptor}, 1 seed, 200k iterations (a deliberate over-run to detect a plateau)"],
   ["pilot summary", "notebooks/41_pilot_summary.ipynb", "wall time, peak VRAM and plateau iteration (pre-declared rule: 3-point running median; first point after which no later point improves by more than 5% (2% strict) of the total drop)"]], [3*cm, 4.6*cm, 9.7*cm])
H2("Design decisions to be aware of")
B(["The gate multiplies the <i>whole</i> condition embedding (ligand + donor + cell type pooled), not the ligand-only latent — a simplification, because CellFlow pools all covariates before the after-pool MLP.",
   "Gate input for <font name='DVM'>receptor</font>: binding row of the ligand × receptor abundance of the (donor, cell type), 120-d. <font name='DVM'>receptor_permuted</font>: cell-type→abundance map shuffled within each donor. <font name='DVM'>mlp_full_basal</font>: binding row + full 2,000-gene PBS mean through a wider MLP — my interpretation; the brief did not define its input.",
   "The 'binding' pert_rep is the 120-d signed binding row of B, not the 90×90 distance matrix used in notebook 22.",
   "<b>Pending Stage C requirement:</b> the PCA was fitted excluding only IL-2 and IFN-γ. GM-CSF, TNF-α, IL-4 and EGF were in its fit — an unsupervised leak that should be removed (refit) or declared before the 72-run experiment."])

H1("Part 10 — Cross-cutting caveats and open issues")
B(["<b>One trained baseline.</b> The only completed trained models are 06 and 08's arms B and C; all use a single seed, 3 test donors and a single held-out cytokine (IL-15) <i>that was in training for one donor</i>. No variance estimate exists. These do not establish zero-shot ligand generalisation.",
   "<b>Well-level null.</b> Any small energy distance (&lt; 2× the well floor) must be read against the well null (09/20), which 06/08/31 do not apply. 06/08 report distances in a 30-PC space, 09 in 50 PCs, 30/31 in 100 PCs — numbers across notebooks are <b>not comparable</b>.",
   "<b>Energy-distance bias.</b> Comparisons within a donor × cell type must use identical n (V-statistic bias ≈ 1/n); notebooks 31 and 40 cap at 1,000 per side; 06/08 use 2,000 for validation.",
   "<b>Reliability.</b> Only 46/90 cytokines have a reproducible response; 44 look like pure noise at the well null, so most of the 'zero-shot to 6 ligands' targets (IL-4, EGF, TNF-α, GM-CSF...) differ greatly in how much true signal exists (EGF, TNF-α are null in 09).",
   "<b>Receptor annotation gaps.</b> 5 cytokines without annotation; PSPN NaN; Noggin/Decorin evidence B/C; decoy receptors excluded by choice.",
   "<b>Unfilled conclusions.</b> Notebook 08's conclusion cell is empty; notes.md still lists the 08 run as 'in training' although its CSVs exist; interpretation in this report is from the CSVs.",
   "<b>Infrastructure.</b> 01–09 ran on the MDC cluster (A40 GPUs); everything from 20 on runs on the Cyfronet cluster (CPU in practice); no GPU job has run on the new cluster yet, GPU queue is heavily contended, and the CUDA install there is unverified.",
   "<b>Version control.</b> Only the initial commit exists; DATA.md, HARDWARE.md, RECEPTORS.md, SETUP.md, notebooks 20–32/40/41, scripts, the fork, tests and all new data files are untracked."])
H2("File map")
T([["path", "contents"], ["notebooks/01–09, predicting_cytokines", "setup, baseline, embeddings, controls, receptor competence; <font name='DVM'>predicting_cytokines</font> is the unmodified CellFlow tutorial stub"],
   ["notebooks/20–22, 30–32, 40, 41", "response matrix, receptor layer, representation screen, training set, baselines, Stage B"], ["scripts/build/nb*.py", "generators for notebooks 20–22, 30–32, 40, 41 and the verdict texts"],
   ["models/", "06_baseline_CellFlow.pkl, metric CSVs (06, 08, 09), figures (07, 20, 21, 22)"], ["data/", "reliability.csv, response_matrix.h5ad, binding/competence parquet, representation_screen*.csv, agonist_antagonist_panel.csv, baselines_train90.csv, pbmc_train90.h5ad (+coverage)"],
   ["cellflow_fork/, tests/", "Stage B code and unit tests"], ["reports/", "this report and its builder (scripts/build_report.py)"]], [5*cm, 12.3*cm])

def footer(c, d):
    c.saveState(); c.setFont("DV", 7); c.setFillColor(colors.HexColor("#777777")); c.drawString(1.8*cm, 1.1*cm, "CellFlow cytokine project — experiment report"); c.drawRightString(A4[0]-1.8*cm, 1.1*cm, f"page {d.page}"); c.restoreState()
doc = SimpleDocTemplate("reports/cellflow_experiments_report.pdf", pagesize=A4, leftMargin=1.8*cm, rightMargin=1.8*cm, topMargin=1.7*cm, bottomMargin=1.8*cm, title="CellFlow cytokine project — experiment report", author="Claude Code")
doc.build(E, onFirstPage=footer, onLaterPages=footer); print("wrote reports/cellflow_experiments_report.pdf")
