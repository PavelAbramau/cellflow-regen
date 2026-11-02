def add_partB(md, code):
    md("""# Part B — agonist / antagonist panel (CPU, no training)

**Question.** Does a *structural* representation (ESM2) confuse an agonist with its competitive antagonist, which share a receptor and a fold but have opposite function?

**Pairs.** Antagonist pairs: `IL-1-beta / IL-1Ra` (IL1R1) and `IL-36-alpha / IL-36Ra` (IL1RL2). Positive controls, which should be close in every space: `IL-2 / IL-15` (IL2RB + IL2RG) and `M-CSF / IL-34` (CSF1R).

**Prediction (stated before looking).** Antagonist pairs: NEAR in ESM2, FAR in signed binding (sign flip), FAR in response. Positive controls: near in all three.

**Per pair and per space (every Part A space + the observed response distance `D_response`)** — (1) cosine distance between the members; (2) rank of the partner among the 89 other cytokines by distance, from each member's side (1 = nearest; ties get the average rank); (3) percentile of the pair distance in the full pairwise distribution (4,005 pairs; 0 = closest pair of all, 100 = farthest).

**Caveat checked first.** The response distance of a pair is only interpretable if BOTH members have a reproducible response (split-half r >= 0.5, notebook 20). Pairs failing this are reported and flagged, not dropped.""")
    code("""# self-contained: rebuild the Part A distance matrices if this block is run on its own (identical recipe, no sentence-transformers needed)
import numpy as np, pandas as pd, anndata as ad, glob, scipy.sparse as sp, matplotlib.pyplot as plt
from scipy.stats import rankdata
if "reps" not in globals():
    def cosd(M):
        M=np.asarray(M,float); nn=np.linalg.norm(M,axis=1,keepdims=True); Mn=M/np.where(nn==0,1,nn); D=np.clip(1-Mn@Mn.T,0,2); np.fill_diagonal(D,0); return D
    A=ad.read_h5ad("../data/response_matrix.h5ad"); LIG=list(A.obs_names); n_all=len(LIG)
    R=A.X if not sp.issparse(A.X) else A.X.toarray(); D_resp_all=cosd(R)
    f=glob.glob("/users/scratch1/pavelabramau/data_pavel/cellflow/hf_cache/hub/datasets--theislab--cellflow-datasets/snapshots/*/pbmc_parse.h5ad")[0]
    esm=dict(ad.read_h5ad(f,backed="r").uns["esm2_embeddings"]); E=np.vstack([np.asarray(esm[c],float) for c in LIG])
    F=pd.get_dummies(A.obs["family"].astype(str).values).values.astype(float)
    reps={"esm2_raw":cosd(E),"esm2_centred":cosd(E-E.mean(0)),"binding":np.load("../data/D_binding_signed.npy"),"family_onehot":cosd(F)}
    const=np.ones((n_all,n_all)); np.fill_diagonal(const,0); reps["constant"]=const
    reps["random_gauss"]=cosd(np.random.default_rng(0).normal(size=(n_all,64)))
    reps["uniprot_text"]=cosd(np.load("../data/uniprot_text_embedding.npy"))
REL=pd.read_csv("../data/reliability.csv").set_index("cytokine")
SPACES=dict(reps); SPACES["RESPONSE (observed)"]=D_resp_all
PAIRS=[("IL-1-beta","IL-1Ra","antagonist","IL1R1"),("IL-36-alpha","IL-36Ra","antagonist","IL1RL2"),("IL-2","IL-15","positive control","IL2RB+IL2RG"),("M-CSF","IL-34","positive control","CSF1R")]
assert all(a in LIG and b in LIG for a,b,_,_ in PAIRS)
I,J=np.triu_indices(len(LIG),1)""")
    code("""rows=[]
for a,b,kind,rec in PAIRS:
    ia,ib=LIG.index(a),LIG.index(b)
    for sp_,D in SPACES.items():
        d=D[ia,ib]; allp=D[I,J]
        ra=rankdata(np.delete(D[ia],[ia]),method="average")[ (ib if ib<ia else ib-1) ]   # rank of b among the 89 others of a
        rb=rankdata(np.delete(D[ib],[ib]),method="average")[ (ia if ia<ib else ia-1) ]
        rows.append(dict(pair=f"{a} / {b}",kind=kind,receptor=rec,space=sp_,distance=d,rank_b_from_a=ra,rank_a_from_b=rb,
                         percentile=100*(np.sum(allp<d)+0.5*np.sum(allp==d))/len(allp),n_tied_at_d=int(np.sum(allp==d))))
T=pd.DataFrame(rows)
rel=pd.DataFrame([dict(pair=f"{a} / {b}",member_a=a,r_a=REL.loc[a,"r_split_half"],member_b=b,r_b=REL.loc[b,"r_split_half"],
        both_reliable=bool(REL.loc[a,"reliable"] and REL.loc[b,"reliable"]),category_a=REL.loc[a,"category"],category_b=REL.loc[b,"category"]) for a,b,_,_ in PAIRS])
print("RELIABILITY CHECK (split-half r >= 0.5 needed for BOTH members):"); print(rel.to_string(index=False))
bad=[p for p,ok in zip(rel.pair,rel.both_reliable) if not ok]
print("\\n-> response distance NOT interpretable for:",bad if bad else "none")
T["response_interpretable"]=T.pair.map(dict(zip(rel.pair,rel.both_reliable)))
T.to_csv("../data/agonist_antagonist_panel.csv",index=False)
# one small table: rows = pairs, columns = spaces; cell = distance | rank(a->b, b->a) | percentile
W=T.assign(cell=lambda x:x.apply(lambda r:f"{r.distance:.3f} | r{r.rank_b_from_a:.0f},{r.rank_a_from_b:.0f} | p{r.percentile:.0f}",axis=1)).pivot(index="pair",columns="space",values="cell")
W=W.loc[[f"{a} / {b}" for a,b,_,_ in PAIRS],list(SPACES)]; W.insert(0,"response ok?",[ "yes" if ok else "NO" for ok in rel.both_reliable]); W.insert(0,"kind",[k for _,_,k,_ in PAIRS])
print("\\ncell = cosine distance | rank of partner among 89 others (a->b, b->a) | percentile in all 4,005 pairs (0 = closest)"); print(W.to_string())""")
    code("""spaces=list(SPACES); fig,ax=plt.subplots(figsize=(11,3.8))
cols={"antagonist":"#c0392b","positive control":"#2471a3"}; mk=["o","s","^","D"]
for k,(a,b,kind,rec) in enumerate(PAIRS):
    t=T[T.pair==f"{a} / {b}"].set_index("space").loc[spaces]; ok=bool(rel.both_reliable[k])
    xs=np.arange(len(spaces))+(k-1.5)*0.14
    for x,s in zip(xs,spaces):
        hollow=(s=="RESPONSE (observed)" and not ok)
        ax.scatter(x,t.loc[s,"percentile"],marker=mk[k],s=55,facecolors="none" if hollow else cols[kind],edgecolors=cols[kind],linewidths=1.4,label=f"{a} / {b}" if s==spaces[0] else None)
ax.set_xticks(range(len(spaces))); ax.set_xticklabels(spaces,rotation=25,ha="right",fontsize=8); ax.set_ylabel("percentile of pair distance\\n(0 = closest of 4,005 pairs)")
ax.axhline(50,color="grey",lw=.6,ls=":"); ax.set_ylim(-3,103); ax.legend(fontsize=7,ncol=4,loc="upper center",bbox_to_anchor=(.5,1.16))
fig.suptitle("Agonist/antagonist panel: red = antagonist pairs, blue = positive controls.  Hollow marker = response distance NOT interpretable (a member has split-half r < 0.5).",fontsize=8,y=0.995)
plt.tight_layout(); plt.savefig("../models/22_agonist_antagonist.png",dpi=130); plt.show()""")
