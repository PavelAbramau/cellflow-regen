import nbformat as nbf
C=[]
md=lambda s:C.append(nbf.v4.new_markdown_cell(s)); code=lambda s:C.append(nbf.v4.new_code_cell(s))
md("""# 30 — Build `data/pbmc_train90.h5ad` (CPU only)

All 90 cytokines, 4 donors, **cap 2,000 cells per (donor, cytokine, cell_type)** (random, seed 0), **PBS in full** for those donors.
Donors are the MDC-work recipe: `Donor8` + the 3 donors with the most PBS cells (`Donor1, Donor12, Donor2`), kept for continuity.

* `X` = log1p(CP10k over the 2,000 HVGs in X), same convention as notebooks 07/20; raw counts in `layers['counts']`.
* `obsm['X_pca']` = 100 PCs. PCA is **fitted on <= 200k random cells that EXCLUDE IL-2 and IFN-gamma** (the pilot's held-out ligands), then applied to every cell.
  Components/mean in `uns['pca']`. The same PCA is used by every arm and baseline.
* `uns['esm2_embeddings']`, `uns['donor_embeddings']` copied from the source file.""")
code("""import glob, time, numpy as np, pandas as pd, anndata as ad, h5py, scipy.sparse as sp
from sklearn.decomposition import PCA
DONORS=["Donor8","Donor1","Donor12","Donor2"]; CAP=2000; HELD=["IL-2","IFN-gamma"]; OUT="../data/pbmc_train90.h5ad"
SRC=glob.glob("../data/cellflow/hf_cache/hub/datasets--theislab--cellflow-datasets/snapshots/*/pbmc_parse.h5ad")[0]
src=ad.read_h5ad(SRC,backed="r"); obs=src.obs.copy(); uns=dict(src.uns); var=src.var.copy(); n=len(obs)
for c in ["donor","cytokine","cell_type"]: obs[c]=obs[c].astype(str)
print(n,"cells in source")
rng=np.random.default_rng(0)
cand=np.where(obs.donor.isin(DONORS).values)[0]
perm=rng.permutation(cand)
sub=obs.iloc[perm][["donor","cytokine","cell_type"]]
rank=sub.groupby(["donor","cytokine","cell_type"]).cumcount().values
keep=perm[(sub.cytokine.values=="PBS")|(rank<CAP)]
keep=np.sort(keep); print("selected",len(keep),"cells")""")
code("""t0=time.time(); isel=np.zeros(n,bool); isel[keep]=True; G=src.n_vars; pieces=[]
with h5py.File(SRC) as f:
    ip=f["X/indptr"][:]; D=f["X/data"]; I=f["X/indices"]
    for s in range(0,n,500_000):
        e=min(n,s+500_000); m=isel[s:e]
        if not m.any(): continue
        lo,hi=ip[s],ip[e]
        X=sp.csr_matrix((D[lo:hi].astype(np.float32),I[lo:hi],ip[s:e+1]-lo),shape=(e-s,G))
        pieces.append(X[np.where(m)[0]])
counts=sp.vstack(pieces).tocsr(); del pieces
lib=np.asarray(counts.sum(1)).ravel(); Xn=sp.diags((1e4/np.maximum(lib,1)).astype(np.float32))@counts; Xn.data=np.log1p(Xn.data); Xn=Xn.tocsr()
print("read+normalise %.0fs"%(time.time()-t0), counts.shape, "max count",counts.max())""")
code("""o=obs.iloc[keep].copy(); o["is_control"]=(o.cytokine=="PBS").values; o["condition"]=o.donor+"_"+o.cytokine
o["cytokine"]=pd.Categorical(o.cytokine); 
a=ad.AnnData(X=Xn,obs=o,var=var); a.layers["counts"]=counts
a.uns["esm2_embeddings"]={k:np.asarray(v) for k,v in uns["esm2_embeddings"].items()}
a.uns["donor_embeddings"]={k:np.asarray(v) for k,v in uns["donor_embeddings"].items()}
# PCA fitted on cells excluding the pilot's held-out ligands
fit_idx=np.where(~a.obs.cytokine.astype(str).isin(HELD).values)[0]; fit_idx=np.random.default_rng(0).choice(fit_idx,size=min(200_000,len(fit_idx)),replace=False)
pca=PCA(n_components=100,svd_solver="randomized",random_state=0).fit(a.X[np.sort(fit_idx)].toarray())
Z=np.vstack([pca.transform(a.X[s:s+100_000].toarray()) for s in range(0,a.n_obs,100_000)]).astype(np.float32)
a.obsm["X_pca"]=Z; a.uns["pca"]=dict(components=pca.components_.astype(np.float32),mean=pca.mean_.astype(np.float32),explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),fit_excludes=HELD,n_fit_cells=len(fit_idx))
a.uns["recipe"]=dict(donors=DONORS,cap_per_donor_cytokine_celltype=CAP,pbs="full",seed=0,normalisation="log1p(CP10k over 2000 HVGs)",source=SRC)
a.write_h5ad(OUT,compression="gzip"); print(a)
print("PCA var explained, 100 PCs: %.3f"%pca.explained_variance_ratio_.sum())""")
md("## Coverage report")
code("""obs_=a.obs.copy(); obs_["cytokine"]=obs_.cytokine.astype(str)
print("cells:",a.n_obs,"| PBS:",int(obs_.is_control.sum()),"| ligand cells:",int((~obs_.is_control).sum()))
print("\\ncells per donor:\\n",obs_.donor.value_counts().to_string())
lig=obs_[~obs_.is_control]
g=lig.groupby(["donor","cytokine","cell_type"]).size()
full=g.reindex(pd.MultiIndex.from_product([DONORS,sorted(lig.cytokine.unique()),sorted(obs_.cell_type.unique())],names=g.index.names)).fillna(0)
print("\\nligands:",lig.cytokine.nunique(),"| (donor,cytokine,cell_type) cells: groups possible",len(full),"| non-empty",int((full>0).sum()),"| >=20 cells",int((full>=20).sum()),"| at cap",int((full>=CAP).sum()))
print("\\n(donor,ligand) pairs present:",int((lig.groupby(["donor","cytokine"]).size()>0).sum()),"of",4*90)
pl=lig.groupby("cytokine").size(); print("cells per ligand: min %d median %d max %d"%(pl.min(),pl.median(),pl.max())); print(pl.sort_values().head(5).to_string())
ctc=(full>=20).groupby(level=2).sum(); ctc=pd.DataFrame({"groups>=20":ctc,"of":4*90,"frac":(ctc/(4*90)).round(2)}); print("\\nper cell type coverage:\\n",ctc.to_string())
pbs=obs_[obs_.is_control].groupby(["donor","cell_type"]).size().unstack(0); print("\\nPBS cells per donor x cell_type:\\n",pbs.to_string())
cov=full.rename("n_cells").reset_index(); cov.to_csv("../data/pbmc_train90_coverage.csv",index=False)""")
nb=nbf.v4.new_notebook(); nb.cells=C; nb.metadata["kernelspec"]={"name":"cellflow","display_name":"cellflow","language":"python"}
nbf.write(nb,"notebooks/30_build_train90.ipynb")
