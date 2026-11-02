import nbformat as nbf
C=[]
md=lambda s:C.append(nbf.v4.new_markdown_cell(s)); code=lambda s,tags=None:C.append(nbf.v4.new_code_cell(s,metadata={"tags":tags} if tags else {}))
md("""# 40 — One zero-shot run (parameterised; execute with papermill, GPU)

Trains CellFlow on **all ligands except `held_out`** (removed from ALL donors: true zero-shot), 4 donors, PBS in full, in the 100-PC space of
`data/pbmc_train90_slim.h5ad`. `(donor, cell_type)` is the condition split (`split_covariates`), so competence can vary per cell type.
Arm = `pert_rep` x `gate` (see `cellflow_fork/cellflow/arms.py`). Validation energy distance on the held-out ligand is logged every `valid_freq` iterations.

Outputs in `<out_root>/<tag>/<run>/`: `val_log.csv` (iteration, wall time, held-out e-distance), `run.json` (wall time, peak VRAM),
`preds.npz` (final predicted X_pca per `donor|cell_type|ligand`), `final_metrics.csv`.""")
code('''held_out = "IL-2"          # one ligand, removed from all donors
pert_rep = "esm2"          # constant | esm2 | binding | binding_esm2 | family_onehot
gate = "none"              # none | receptor | receptor_permuted | mlp_full_basal
seed = 0
num_iterations = 200000
valid_freq = 5000
batch_size = 1024
tag = "pilot"
out_root = "../results/runs"
data_dir = "../data"
min_cells = 20             # groups with fewer held-out or PBS cells are not scored
n_cap = 1000               # cells per side in every comparison (same n everywhere)
require_gpu = True''',["parameters"])
code('''import os, sys, time, json, functools, warnings, subprocess
ROOT = os.path.abspath("..")
sys.path.insert(0, os.path.join(ROOT, "cellflow_fork"))
import numpy as np, pandas as pd, anndata as ad, jax, optax, flax.linen as nn
import cellflow
assert cellflow.__file__.startswith(os.path.join(ROOT, "cellflow_fork")), cellflow.__file__   # the fork, not site-packages
from cellflow.model import CellFlow
from cellflow.utils import match_linear
from cellflow.training import Metrics, ComputationCallback
from cellflow.metrics import compute_e_distance
from cellflow.arms import ArmConfig, prepare_arm
warnings.simplefilter("ignore")
if require_gpu: assert jax.default_backend() == "gpu", jax.devices()
cfg = ArmConfig(pert_rep, gate); run = f"{held_out}__{pert_rep}__{gate}__s{seed}"
out = os.path.join(out_root, tag, run); os.makedirs(out, exist_ok=True)
print(run, jax.devices(), "SLURM_JOB_ID", os.environ.get("SLURM_JOB_ID"))''')
code('''a = ad.read_h5ad(f"{data_dir}/pbmc_train90_slim.h5ad")
for c in ["donor","cell_type","cytokine"]: a.obs[c] = a.obs[c].astype(str)
a.obs["is_control"] = a.obs["is_control"].astype(bool)
assert held_out in set(a.obs.cytokine)
pdk, pmk = prepare_arm(a, cfg, data_dir, seed=seed, basal_path=f"{data_dir}/basal_means_train90.npz")
is_held = (a.obs.cytokine == held_out).to_numpy()
train = a[~is_held].copy(); train.uns = a.uns
print("train cells", train.n_obs, "| held-out cells (all donors)", int(is_held.sum()), "| held-out ligand in train:", held_out in set(train.obs.cytokine))
assert held_out not in set(train.obs.cytokine)          # zero-shot: absent from ALL donors
# validation = held-out ligand + PBS, same cap n per (donor, cell_type); groups with too few cells dropped
rng = np.random.default_rng(0); o = a.obs; keep = []
for (d, c), g in o.groupby(["donor","cell_type"]):
    t = g.index[(g.cytokine == held_out).to_numpy()]; p = g.index[g.is_control.to_numpy()]
    if len(t) < min_cells or len(p) < min_cells: continue
    keep += list(rng.choice(t, min(len(t), n_cap), replace=False)) + list(rng.choice(p, min(len(p), n_cap), replace=False))
val = a[np.isin(a.obs_names, keep)].copy(); val.uns = a.uns
print("validation cells", val.n_obs, "groups", val.obs[["donor","cell_type"]].drop_duplicates().shape[0])''')
code('''cf = CellFlow(train, solver="otfm"); cf.prepare_data(**pdk)
cf.prepare_validation_data(val, name="heldout", n_conditions_on_log_iteration=None, n_conditions_on_train_end=None)
cf.prepare_model(
    condition_mode="deterministic", regularization=0.0, pooling="attention_token", pooling_kwargs={},
    layers_after_pool={"layer_type": "mlp", "dims": [1024, 1024], "dropout_rate": 0.0},
    condition_embedding_dim=256, cond_output_dropout=0.9, pool_sample_covariates=True,
    time_freqs=1024, time_encoder_dims=[1024,1024,1024], hidden_dims=[2048,2048,2048], decoder_dims=[4096,4096,4096],
    conditioning="concatenation", probability_path={"constant_noise": 0.5},
    match_fn=functools.partial(match_linear, epsilon=0.5, tau_a=1.0, tau_b=1.0),
    optimizer=optax.MultiSteps(optax.adam(5e-5), 20), seed=seed, **pmk)
print("conditions in training data:", {k: v.shape for k, v in cf.train_data.condition_data.items()})''')
code('''class Clock(ComputationCallback):
    """records wall time at each validation point (the e-distance itself comes from `Metrics`)"""
    def __init__(self): self.t0 = None; self.wall = []
    def on_train_begin(self, *a, **k): self.t0 = time.perf_counter()
    def on_log_iteration(self, *a, **k): self.wall.append(time.perf_counter() - self.t0); return {"wall_s": self.wall[-1]}
    def on_train_end(self, *a, **k): return {}
clock = Clock()
t0 = time.perf_counter()
cf.train(num_iterations=num_iterations, batch_size=batch_size, valid_freq=valid_freq,
         callbacks=[Metrics(metrics=["e_distance"]), clock], seed=seed)
train_wall_s = time.perf_counter() - t0
ms = jax.devices()[0].memory_stats() or {}
peak_gb = ms.get("peak_bytes_in_use", float("nan")) / 1e9
smi = subprocess.run(["nvidia-smi","--query-gpu=name,memory.used,memory.total","--format=csv,noheader"],capture_output=True,text=True).stdout.strip()
print(f"train wall {train_wall_s:.0f}s | jax peak {peak_gb:.2f} GB | nvidia-smi {smi}")''')
code('''logs = cf.trainer.training_logs; ek = [k for k in logs if k.endswith("e_distance_mean")][0]
vl = pd.DataFrame({"iteration": (np.arange(len(logs[ek])) + 1) * valid_freq, "val_e_distance": logs[ek], "wall_s": clock.wall[:len(logs[ek])],
                   "train_loss": [np.mean(logs["loss"][i*valid_freq:(i+1)*valid_freq]) for i in range(len(logs[ek]))]})
vl.to_csv(f"{out}/val_log.csv", index=False); print(vl.to_string(index=False))''')
code('''# final held-out predictions per (donor, cell_type), then metrics with identical n (<= n_cap) on every side
ctrl = val[val.obs.is_control.to_numpy()]
cov = val[~val.obs.is_control.to_numpy()].obs.drop_duplicates(["donor","cell_type","cytokine"]).copy()
cov["gid"] = cov.donor + "|" + cov.cell_type + "|" + cov.cytokine
preds = cf.predict(adata=ctrl, sample_rep="X_pca", condition_id_key="gid", covariate_data=cov)
Z = val.obsm["X_pca"]; rows = []; P = {}
for gid, arr in preds.items():
    d, c, l = gid.split("|"); arr = np.squeeze(np.asarray(arr)); P[gid] = arr.astype(np.float32)
    m = ((val.obs.donor == d) & (val.obs.cell_type == c)).to_numpy()
    t = Z[m & (val.obs.cytokine == held_out).to_numpy()]; p = Z[m & val.obs.is_control.to_numpy()]
    n = min(len(t), len(p), len(arr))
    r = np.random.default_rng(0); f = lambda x: x[r.choice(len(x), n, replace=False)]
    rows.append(dict(ligand=held_out, donor=d, cell_type=c, n=n, e_arm=float(compute_e_distance(f(t), f(arr))), e_control_mean=float(compute_e_distance(f(t), f(p)))))
fm = pd.DataFrame(rows).assign(pert_rep=pert_rep, gate=gate, seed=seed); fm.to_csv(f"{out}/final_metrics.csv", index=False)
np.savez_compressed(f"{out}/preds.npz", **{k.replace("|", "__"): v for k, v in P.items()})
print(fm[["e_arm","e_control_mean"]].mean().round(3).to_dict(), "| arm beats control in", int((fm.e_arm < fm.e_control_mean).sum()), "of", len(fm), "groups")''')
code('''json.dump(dict(run=run, held_out=held_out, pert_rep=pert_rep, gate=gate, seed=seed, num_iterations=num_iterations, valid_freq=valid_freq,
               batch_size=batch_size, train_cells=int(train.n_obs), train_wall_s=train_wall_s, peak_vram_gb_jax=peak_gb, nvidia_smi=smi,
               gpu=jax.devices()[0].device_kind, slurm_job=os.environ.get("SLURM_JOB_ID"), mean_e_arm=float(fm.e_arm.mean()),
               mean_e_control=float(fm.e_control_mean.mean())), open(f"{out}/run.json", "w"), indent=1)''')
nb=nbf.v4.new_notebook(); nb.cells=C; nb.metadata["kernelspec"]={"name":"cellflow","display_name":"cellflow","language":"python"}
nbf.write(nb,"notebooks/40_run_zeroshot.ipynb")
