import nbformat as nbf, sys
verdict = open("scripts/build/nb22_verdict.md").read() if len(sys.argv)>1 else "**Verdict:** (filled after first run)"
C=[]
md=lambda s:C.append(nbf.v4.new_markdown_cell(s)); code=lambda s:C.append(nbf.v4.new_code_cell(s))
md("# 22 — Representation screen: does any condition representation predict response geometry?\n\n"+verdict+"""

CPU only, no training. Decides whether Stage B happens.

**Design.** Candidate representations → 90×90 cosine-distance matrices, each compared with D_response (notebook 20; cosine distance of response vectors, 14 cell types × 2,000 genes).
* `esm2_raw`: `uns['esm2_embeddings']` (2560-d, not L2-normalised) ; `esm2_centred`: same minus the mean over the 90 cytokines ; `binding`: signed D_binding (notebook 21); `family_onehot`: `cytokine_family` one-hot; `uniprot_text`: sentence embedding (all-MiniLM-L6-v2, mean over ≤200-word chunks) of the UniProt *Function* field (human, reviewed); `constant`: all off-diagonal distances 1 (control; rank correlation undefined → reported as NaN with p = 1); `random_gauss`: 64-d Gaussian vectors (seed 0; extra calibration control).
* **NicheNet: not used** (neither here nor as a candidate), so it is not a role in this screen; notebook 21 annotations are OmniPath only.
* Statistics: (i) Spearman ρ between the flattened upper triangles. (ii) **Mantel permutation test**: object labels of the candidate matrix are permuted jointly on rows and columns (10,000 draws, seed 0, one-sided "greater", p = (1+#≥obs)/(1+10,000)); the flattened vector is never shuffled and no parametric p-value is used. (iii) **Distance correlation** computed from the two distance matrices (double-centred, V-statistic — checked below against `dcor.distance_correlation` on Euclidean data) with its own label-permutation test (same 10,000 permutations).
* Subsets: (a) the **reliable responders** (split-half r ≥ 0.5 → 46 cytokines in this data; the brief expected ~40) and (b) all 90. Also per cell type (response vector restricted to that cell type's 2,000 genes).
* Note: the 5 cytokines without an annotated receptor have an all-zero binding row; their binding distance to everything is fixed at 1.0.""")
code("""import numpy as np, pandas as pd, anndata as ad, dcor, json, os, time, warnings, matplotlib.pyplot as plt, scipy.sparse as sp
from scipy.stats import rankdata, spearmanr
warnings.simplefilter("ignore"); pd.set_option("display.width",220,"display.max_rows",300)
A=ad.read_h5ad("../data/response_matrix.h5ad"); LIG=list(A.obs_names); n_all=len(LIG)
rel=A.obs["reliable"].values.astype(bool); ct_of=A.var["cell_type"].values; CTS=list(A.uns["cell_types_kept"])
def cosd(M):
    M=np.asarray(M,float); nn=np.linalg.norm(M,axis=1,keepdims=True); Mn=M/np.where(nn==0,1,nn); D=np.clip(1-Mn@Mn.T,0,2); np.fill_diagonal(D,0); return D
R=A.X if not sp.issparse(A.X) else A.X.toarray()
D_resp_all=cosd(R); assert np.allclose(D_resp_all,A.obsp["D_response_all"],atol=1e-5)
D_resp_ct={ct:cosd(R[:,ct_of==ct]) for ct in CTS}
print("objects:",n_all,"reliable:",rel.sum(),"cell types:",len(CTS))""")
md("## Candidate representations")
code("""import glob
f=glob.glob("/users/scratch1/pavelabramau/data_pavel/cellflow/hf_cache/hub/datasets--theislab--cellflow-datasets/snapshots/*/pbmc_parse.h5ad")[0]
esm=dict(ad.read_h5ad(f,backed="r").uns["esm2_embeddings"]); E=np.vstack([np.asarray(esm[c],float) for c in LIG])
print("ESM2",E.shape,"norms",E.min() if False else np.linalg.norm(E,axis=1).round(1).min(),np.linalg.norm(E,axis=1).round(1).max())
fam=A.obs["family"].astype(str).values; F=pd.get_dummies(fam).values.astype(float)
reps={}
reps["esm2_raw"]=cosd(E); reps["esm2_centred"]=cosd(E-E.mean(0))
reps["binding"]=np.load("../data/D_binding_signed.npy"); reps["family_onehot"]=cosd(F)
const=np.ones((n_all,n_all)); np.fill_diagonal(const,0); reps["constant"]=const
reps["random_gauss"]=cosd(np.random.default_rng(0).normal(size=(n_all,64)))
# UniProt text
up=pd.read_csv("../data/uniprot_function.csv").set_index("cytokine").reindex(LIG)
up["function"]=up["function"].fillna("")
print("empty UniProt function:",list(up.index[up["function"].str.len()==0]),"-> embedded from the gene symbol")
from sentence_transformers import SentenceTransformer
st=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",device="cpu")
def chunks(t,g,k=200):
    w=t.split(); return [" ".join(w[i:i+k]) for i in range(0,len(w),k)] or [g]
Tm=[]
for c in LIG:
    ch=chunks(up.loc[c,"function"],up.loc[c,"genes"]); Tm.append(st.encode(ch,normalize_embeddings=True).mean(0))
Tm=np.vstack(Tm); reps["uniprot_text"]=cosd(Tm)
np.save("../data/uniprot_text_embedding.npy",Tm)
for k,v in reps.items(): assert v.shape==(90,90) and np.allclose(v,v.T,atol=1e-6); print(f"{k:14s} median off-diag {np.median(v[np.triu_indices(90,1)]):.3f}")""")
md("## Statistics")
code("""def rnd(df,k=4): return df.round({c:k for c in df.select_dtypes("number").columns})
def dc(D):
    r=D.mean(0,keepdims=True); c=D.mean(1,keepdims=True); return D-r-c+D.mean()
def spear_mantel(Dc,Dr,P,chunk=1000):
    n=len(Dc); I,J=np.triu_indices(n,1); a=Dc[I,J]; b=Dr[I,J]
    if np.ptp(a)==0 or np.ptp(b)==0: return np.nan,1.0
    rb=rankdata(b); rb=(rb-rb.mean()); ra=rankdata(a); RK=np.zeros((n,n)); RK[I,J]=ra; RK=RK+RK.T
    ssa=np.sum((ra-ra.mean())**2); sd=np.sqrt(ssa*np.sum(rb**2))
    obs=np.sum((ra-ra.mean())*rb)/sd
    ge=0
    for s in range(0,len(P),chunk):
        p=P[s:s+chunk]; V=RK[p[:,I],p[:,J]]; r=(V@rb)/sd; ge+=np.sum(r>=obs-1e-12)
    return obs,(1+ge)/(1+len(P))
def dcor_perm(Dc,Dr,P,chunk=250):
    n=len(Dc); Ac=dc(Dc); Bc=dc(Dr); s=np.sqrt(np.sum(Ac**2)*np.sum(Bc**2))
    if s<1e-12: return 0.0,1.0
    obs=np.sum(Ac*Bc)/s; ge=0
    for k in range(0,len(P),chunk):
        p=P[k:k+chunk]; V=Ac[p[:,:,None],p[:,None,:]]; r=(V*Bc).sum((1,2))/s; ge+=np.sum(r>=obs-1e-12)
    return float(np.sqrt(max(obs,0))),(1+ge)/(1+len(P))
# validate against dcor on Euclidean data
rng=np.random.default_rng(1); X1=rng.normal(size=(40,5)); X2=X1[:,:2]@rng.normal(size=(2,3))+rng.normal(size=(40,3))
from scipy.spatial.distance import cdist
d1,d2=cdist(X1,X1),cdist(X2,X2); mine=np.sqrt(max(np.sum(dc(d1)*dc(d2))/np.sqrt(np.sum(dc(d1)**2)*np.sum(dc(d2)**2)),0))
print("dCor mine %.6f vs dcor package %.6f"%(mine,dcor.distance_correlation(X1,X2))); assert abs(mine-dcor.distance_correlation(X1,X2))<1e-6
NPERM=10000; rng=np.random.default_rng(0); PERMS={}
def perms(n):
    if n not in PERMS: PERMS[n]=np.vstack([rng.permutation(n) for _ in range(NPERM)])
    return PERMS[n]
def screen(Dr,idx,tag,ct):
    out=[]; P=perms(len(idx)); Drs=Dr[np.ix_(idx,idx)]
    for name,Dcand in reps.items():
        Dcs=Dcand[np.ix_(idx,idx)]; rho,pm=spear_mantel(Dcs,Drs,P); dco,pd_=dcor_perm(Dcs,Drs,P)
        out.append(dict(representation=name,subset=tag,cell_type=ct,rho=rho,mantel_p=pm,dcor=dco,dcor_p=pd_,n=len(idx)))
    return out
SUBS={"reliable":np.where(rel)[0],"all90":np.arange(n_all)}
t0=time.time(); rows=[]
for tag,idx in SUBS.items(): rows+=screen(D_resp_all,idx,tag,"all (concatenated)")
print("main tables done in %.0fs"%(time.time()-t0))
main=pd.DataFrame(rows)
for tag in SUBS: print("\\n===",tag,"(n=%d)"%len(SUBS[tag])); print(rnd(main[main.subset==tag].drop(columns=["subset","cell_type"])).to_string(index=False))""")
code("""t0=time.time()
for ct in CTS:
    for tag,idx in SUBS.items(): rows+=screen(D_resp_ct[ct],idx,tag,ct)
res=pd.DataFrame(rows); res.to_csv("../data/representation_screen.csv",index=False); print("per-cell-type done in %.0fs"%(time.time()-t0))
for tag in SUBS:
    t=res[(res.subset==tag)&(res.cell_type!="all (concatenated)")]
    print("\\n=== rho per cell type —",tag); print(t.pivot(index="cell_type",columns="representation",values="rho").round(3).to_string())
    print("--- Mantel p"); print(t.pivot(index="cell_type",columns="representation",values="mantel_p").round(4).to_string())""")
md("## Does binding beat ESM2? Object-level bootstrap of the rho difference")
code("""# Two separate p-values do not answer "is A better than B". Resample OBJECTS (cytokines) with replacement, drop pairs of a cytokine with itself,
# recompute both Spearman rhos on the same resample, and look at the difference. 2,000 resamples, seed 0.
from scipy.stats import spearmanr
def boot_diff(a,b,Dr,idx,B=2000,seed=0):
    r=np.random.default_rng(seed); n=len(idx); I,J=np.triu_indices(n,1); out=[]
    for _ in range(B):
        s=idx[r.integers(0,n,n)]; m=s[I]!=s[J]; i,j=s[I][m],s[J][m]
        y=Dr[i,j]; out.append(spearmanr(a[i,j],y)[0]-spearmanr(b[i,j],y)[0])
    out=np.array(out); return np.nanmean(out),np.nanpercentile(out,[2.5,97.5]),np.nanmean(out<=0)
brows=[]
for tag,idx in SUBS.items():
    for other in ["esm2_raw","esm2_centred","family_onehot"]:
        m,ci,p1=boot_diff(reps["binding"],reps[other],D_resp_all,idx)
        brows.append(dict(subset=tag,contrast="binding - "+other,mean_diff=m,ci_lo=ci[0],ci_hi=ci[1],frac_boot_le_0=p1))
bd=pd.DataFrame(brows); bd.to_csv("../data/representation_screen_binding_vs_esm2.csv",index=False); print(rnd(bd).to_string(index=False))""")
md("## Figure: D_response vs D_best and vs D_esm2")
code("""mr=res[(res.subset=="reliable")&(res.cell_type=="all (concatenated)")&~res.representation.isin(["constant","random_gauss"])].set_index("representation")
best=mr.rho.idxmax(); print("best representation on the reliable set:",best,mr.loc[best,["rho","mantel_p"]].astype(float).round(4).to_dict())
fig,axs=plt.subplots(2,2,figsize=(10,9))
for r_,(tag,idx) in enumerate(SUBS.items()):
    I,J=np.triu_indices(len(idx),1); y=D_resp_all[np.ix_(idx,idx)][I,J]
    for c_,name in enumerate([best,"esm2_raw"]):
        x=reps[name][np.ix_(idx,idx)][I,J]; a=axs[r_,c_]; a.hexbin(x,y,gridsize=40,cmap="viridis",mincnt=1,bins="log")
        s=res[(res.representation==name)&(res.subset==tag)&(res.cell_type=="all (concatenated)")].iloc[0]
        a.set_title(f"{tag} (n={len(idx)}): {name}\\nrho={s.rho:.3f}, Mantel p={s.mantel_p:.4f}, dCor={s.dcor:.2f} (p={s.dcor_p:.4f})",fontsize=9)
        a.set_xlabel("D_"+name); a.set_ylabel("D_response")
plt.tight_layout(); plt.savefig("../models/22_response_vs_best_vs_esm2.png",dpi=130); plt.show()""")
nb=nbf.v4.new_notebook(); nb.cells=C; nb.metadata["kernelspec"]={"name":"cellflow","display_name":"cellflow","language":"python"}
nbf.write(nb,"notebooks/22_representation_screen.ipynb")
