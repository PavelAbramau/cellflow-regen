import nbformat as nbf
C=[]
md=lambda s:C.append(nbf.v4.new_markdown_cell(s)); code=lambda s:C.append(nbf.v4.new_code_cell(s))
md("""# 32 — Slim training file + basal means (CPU)

* `data/pbmc_train90_slim.h5ad`: obs, `obsm['X_pca']`, `uns` — **no expression matrix** (`X` is empty). Every GPU run loads this (seconds, not minutes);
  the expression matrix in `pbmc_train90.h5ad` is only needed for gene-space metrics.
* `data/basal_means_train90.npz`: mean log1p(CP10k) of PBS cells per (donor, cell_type) over the 2,000 HVGs — the input of the `mlp_full_basal` gate.""")
code("""import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, time
t0=time.time(); a=ad.read_h5ad("../data/pbmc_train90.h5ad"); print(a, "%.0fs"%(time.time()-t0))
o=a.obs.copy()
for c in ["donor","cell_type","cytokine"]: o[c]=o[c].astype(str)
pbs=np.where(o.is_control.values)[0]; X=a.X[pbs].toarray() if sp.issparse(a.X) else a.X[pbs]
key=(o.donor+"|"+o.cell_type).values[pbs]; ks=sorted(set(key)); M=np.stack([X[key==k].mean(0) for k in ks]).astype(np.float32)
np.savez("../data/basal_means_train90.npz",keys=np.array(ks),X=M,genes=np.array(a.var_names)); print("basal means",M.shape)
s=ad.AnnData(X=sp.csr_matrix((a.n_obs,1),dtype=np.float32),obs=a.obs.copy(),obsm={"X_pca":a.obsm["X_pca"].copy()},uns=a.uns)
s.write_h5ad("../data/pbmc_train90_slim.h5ad"); print(s)""")
code("""print(o.cytokine.nunique(),"cytokines (incl PBS);",o.donor.nunique(),"donors;",a.n_obs,"cells")
print(o[~o.is_control.values].groupby(["donor"]).size())
print(sorted(o.cell_type.unique()))""")
nb=nbf.v4.new_notebook(); nb.cells=C; nb.metadata["kernelspec"]={"name":"cellflow","display_name":"cellflow","language":"python"}
nbf.write(nb,"notebooks/32_slim_and_basal.ipynb")
