"""Stream pbmc_parse.h5ad once; accumulate sum of log1p(CP10k) per group.
CP10k is over the 2,000 HVGs in X (library = X row sum), same convention as notebook 07.
Groups A: (donor, cytokine, cell_type, bc1_well)   -> every ligand and PBS, PBS keeps its wells
Groups B: (donor, cell_type, half) for PBS cells, random 50/50 split (seed 0) -> 'random-split' null
Output: data/pseudobulk_log1p_cp10k.npz"""
import glob, sys, numpy as np, pandas as pd, h5py, scipy.sparse as sp, time
OUT = sys.argv[1]
f = glob.glob("/users/scratch1/pavelabramau/data_pavel/cellflow/hf_cache/hub/datasets--theislab--cellflow-datasets/snapshots/*/pbmc_parse.h5ad")[0]
import anndata as ad
a = ad.read_h5ad(f, backed="r"); o = a.obs[["donor","cytokine","cell_type","bc1_well"]].astype(str).reset_index(drop=True); n = len(o)
gene_names = np.array(a.var_names); a.file.close()
kA = o.donor+"|"+o.cytokine+"|"+o.cell_type+"|"+o.bc1_well
ucA, iA = np.unique(kA.values, return_inverse=True)
rng = np.random.default_rng(0); half = rng.integers(0,2,n)
isP = (o.cytokine=="PBS").values
kB = np.where(isP, o.donor+"|"+o.cell_type+"|"+pd.Series(half).astype(str), "")
ucB, iB = np.unique(kB, return_inverse=True)  # '' is group 0 (ignored)
SA = np.zeros((len(ucA), len(gene_names))); CA = np.zeros(len(ucA))
SB = np.zeros((len(ucB), len(gene_names))); CB = np.zeros(len(ucB))
t0=time.time(); CH=400_000
with h5py.File(f) as h:
    ip=h["X/indptr"][:]; D=h["X/data"]; I=h["X/indices"]
    for s in range(0,n,CH):
        e=min(n,s+CH); lo,hi=ip[s],ip[e]
        X=sp.csr_matrix((D[lo:hi].astype(np.float32),I[lo:hi],ip[s:e+1]-lo),shape=(e-s,len(gene_names)))
        lib=np.asarray(X.sum(1)).ravel(); X=sp.diags(1e4/np.maximum(lib,1))@X
        X.data=np.log1p(X.data)
        r=np.arange(e-s)
        GA=sp.csr_matrix((np.ones(e-s),(iA[s:e],r)),shape=(len(ucA),e-s)); SA+=(GA@X).toarray(); CA+=np.asarray(GA.sum(1)).ravel()
        GB=sp.csr_matrix((np.ones(e-s),(iB[s:e],r)),shape=(len(ucB),e-s)); SB+=(GB@X).toarray(); CB+=np.asarray(GB.sum(1)).ravel()
        print(s,e,round(time.time()-t0),flush=True)
np.savez_compressed(OUT,keysA=ucA,sumA=SA,nA=CA,keysB=ucB,sumB=SB,nB=CB,genes=gene_names)
print("done")
