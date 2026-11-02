import nbformat as nbf
C=[]
md=lambda s:C.append(nbf.v4.new_markdown_cell(s)); code=lambda s:C.append(nbf.v4.new_code_cell(s))
md("""# 31 — Analytic baselines for every (held-out ligand, donor, cell_type) (CPU only, no model)

Data: `data/pbmc_train90.h5ad` (notebook 30). Each of the 90 ligands is treated as held out in turn (leave-one-ligand-out), within each (donor, cell_type) with >= 20 cells for the ligand AND for PBS.

* **control_mean baseline:** prediction = the donor's PBS cells of the same cell type (no perturbation effect).
* **cytokine_mean baseline:** prediction = those PBS cells **translated by the mean delta of the OTHER 89 ligands** in the same (donor, cell_type): `delta_bar = mean_{L' != L}(mean_L' - mean_PBS)`, in PCA space (100 PCs) for the e-distance and in log1p(CP10k) gene space for R². The held-out ligand is never part of its own delta_bar.
* Metrics: `cellflow.metrics.compute_e_distance` on `X_pca` (100 PCs) and `compute_r_squared` on 2,000 genes, both against the true cells of the held-out ligand. Each side is subsampled to <= 1,000 cells (seed 0) so e-distance floors are comparable (the V-statistic bias depends on n).
* No well-level null here; see notebook 20 for why one is needed when interpreting small e-distances.""")
code("""import numpy as np, pandas as pd, anndata as ad, time, warnings
from cellflow.metrics import compute_e_distance, compute_r_squared
warnings.simplefilter("ignore"); pd.set_option("display.width",200,"display.max_rows",200)
a=ad.read_h5ad("../data/pbmc_train90.h5ad"); o=a.obs; Z=a.obsm["X_pca"]; X=a.X.tocsr(); MIN=20; NMAX=1000
key=(o.donor.astype(str)+"|"+o.cytokine.astype(str)+"|"+o.cell_type.astype(str)).values
idx=pd.Series(np.arange(a.n_obs)).groupby(key).indices
mZ,mX,N={},{},{}
t0=time.time()
for k,i in idx.items():
    N[k]=len(i)
    if len(i)>=MIN: mZ[k]=Z[i].mean(0); mX[k]=np.asarray(X[i].mean(0)).ravel()
print("group means %.0fs, groups %d, usable %d"%(time.time()-t0,len(idx),len(mZ)))
donors=sorted(o.donor.astype(str).unique()); cts=sorted(o.cell_type.astype(str).unique()); ligs=sorted(set(o.cytokine.astype(str))-{"PBS"})""")
code("""rng=np.random.default_rng(0)
def sub(i): return i if len(i)<=NMAX else rng.choice(i,NMAX,replace=False)
rows=[]; t0=time.time()
for d in donors:
    for c in cts:
        kp=f"{d}|PBS|{c}"
        if kp not in mZ: continue
        valid=[l for l in ligs if f"{d}|{l}|{c}" in mZ]
        if len(valid)<2: continue
        dZ={l:mZ[f"{d}|{l}|{c}"]-mZ[kp] for l in valid}; dX={l:mX[f"{d}|{l}|{c}"]-mX[kp] for l in valid}
        sZ=sum(dZ.values()); sX=sum(dX.values()); k=len(valid)
        ip=sub(idx[kp]); Zp=Z[ip]; Xp=X[ip].toarray()
        for l in valid:
            it=sub(idx[f"{d}|{l}|{c}"]); Zt=Z[it]; Xt=X[it].toarray()
            bZ=(sZ-dZ[l])/(k-1); bX=(sX-dX[l])/(k-1)
            for name,(pz,px) in {"control_mean":(Zp,Xp),"cytokine_mean":(Zp+bZ,Xp+bX)}.items():
                rows.append(dict(ligand=l,donor=d,cell_type=c,baseline=name,n_true=N[f"{d}|{l}|{c}"],n_pbs=N[kp],n_ligands_in_mean=k-1,
                                 e_distance=float(compute_e_distance(Zt,pz)),r_squared=float(compute_r_squared(Xt,px))))
    print(d,"done %.0fs"%(time.time()-t0),flush=True)
B=pd.DataFrame(rows); B.to_csv("../data/baselines_train90.csv",index=False); print(B.shape, B.ligand.nunique(),"ligands")""")
code("""P=B.pivot_table(index=["ligand","donor","cell_type"],columns="baseline",values=["e_distance","r_squared"])
print("overall mean over %d (ligand,donor,cell_type) cases:"%len(P)); print(B.groupby("baseline")[["e_distance","r_squared"]].mean().round(4))
win=(P[("e_distance","cytokine_mean")]<P[("e_distance","control_mean")]); print("cytokine_mean beats control_mean on e-distance in %.1f%% of cases, on R^2 in %.1f%%"%(100*win.mean(),100*(P[("r_squared","cytokine_mean")]>P[("r_squared","control_mean")]).mean()))
for l in ["IL-2","IFN-gamma"]:
    t=B[B.ligand==l]; print("\\n=== held-out",l,"| mean over donors x cell types:"); print(t.groupby("baseline")[["e_distance","r_squared"]].mean().round(4))
    print(t.pivot_table(index="cell_type",columns="baseline",values="e_distance").round(2).to_string())
print("\\nby cell type (mean e-distance, all ligands):"); print(B.pivot_table(index="cell_type",columns="baseline",values="e_distance").round(2).to_string())""")
nb=nbf.v4.new_notebook(); nb.cells=C; nb.metadata["kernelspec"]={"name":"cellflow","display_name":"cellflow","language":"python"}
nbf.write(nb,"notebooks/31_baselines.ipynb")
