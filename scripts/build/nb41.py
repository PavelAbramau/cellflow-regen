import nbformat as nbf
C=[]
md=lambda s:C.append(nbf.v4.new_markdown_cell(s)); code=lambda s,tags=None:C.append(nbf.v4.new_code_cell(s,metadata={"tags":tags} if tags else {}))
md("""# 41 — Pilot summary: wall time, peak VRAM, validation plateau (CPU)

Reads `results/runs/<tag>/*/{run.json,val_log.csv,final_metrics.csv}`. **Plateau rule** (pre-declared): smooth the held-out validation energy distance with a
3-point running median; `best` = its minimum; the plateau iteration is the first validation point from which the smoothed curve never again improves on
`value_there` by more than `tol x (first_value - best)` (tol = 5% headline, 2% strict). One number per run, then the max / median across runs.""")
code('''tag = "pilot"; tol_main = 0.05; tol_strict = 0.02''',["parameters"])
code('''import glob, json, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
runs = []; curves = {}
for f in sorted(glob.glob(f"../results/runs/{tag}/*/run.json")):
    r = json.load(open(f)); d = f.rsplit("/", 1)[0]; v = pd.read_csv(f"{d}/val_log.csv")
    s = v.val_e_distance.rolling(3, center=True, min_periods=1).median().to_numpy(); it = v.iteration.to_numpy(); span = s[0] - s.min()
    def plateau(tol):
        for i in range(len(s)):
            if s[i:].min() >= s[i] - tol * span: return int(it[i])
        return int(it[-1])
    r.update(plateau_5pct=plateau(tol_main), plateau_2pct=plateau(tol_strict), best_val=float(s.min()), best_iter=int(it[s.argmin()]),
             first_val=float(s[0]), last_val=float(s[-1]), s_per_1k_it=r["train_wall_s"] / r["num_iterations"] * 1000)
    runs.append(r); curves[r["run"]] = (it, v.val_e_distance.to_numpy(), s)
R = pd.DataFrame(runs); cols = ["run","train_wall_s","peak_vram_gb_jax","s_per_1k_it","best_iter","best_val","first_val","last_val","plateau_5pct","plateau_2pct","mean_e_arm","mean_e_control"]
print(R[cols].round(3).to_string(index=False))
R.to_csv(f"../results/pilot_summary.csv", index=False)''')
code('''fig, axs = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True)
for ax, lig in zip(axs, sorted(R.held_out.unique())):
    for run, (it, raw, s) in curves.items():
        if run.startswith(lig + "__"): ax.plot(it, raw, lw=1, label=run.split("__", 1)[1].rsplit("__s", 1)[0])
    ax.set_title(f"held-out {lig}"); ax.set_xlabel("iteration"); ax.set_ylabel("validation e-distance (100 PCs)"); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig("../results/pilot_val_curves.png", dpi=130)''')
code('''print("wall time (h): min %.2f median %.2f max %.2f" % tuple(R.train_wall_s.agg(["min","median","max"]) / 3600))
print("peak VRAM (GB, JAX):", R.peak_vram_gb_jax.agg(["min","median","max"]).round(2).to_dict())
print("plateau iteration (5%%): max %d median %d | (2%%): max %d median %d" % (R.plateau_5pct.max(), R.plateau_5pct.median(), R.plateau_2pct.max(), R.plateau_2pct.median()))
print("run length", R.num_iterations.iloc[0], "| any run still improving at the end (best_iter == last)?", bool((R.best_iter == R.num_iterations).any()))''')
nb=nbf.v4.new_notebook(); nb.cells=C; nb.metadata["kernelspec"]={"name":"cellflow","display_name":"cellflow","language":"python"}
nbf.write(nb,"notebooks/41_pilot_summary.ipynb")
