"""Experiment-arm flags for the cytokine zero-shot study (fork-only module).

Two config flags select an arm:

``--pert_rep``  {constant, esm2, binding, binding_esm2, family_onehot}
    What the model is told about the ligand. ``constant`` = one shared vector for every ligand.
``--gate``      {none, receptor, receptor_permuted, mlp_full_basal}
    A scalar MASK in [0, 1) multiplying the condition (ligand) latent; see
    :class:`cellflow.networks._velocity_field.CompetenceGate`.

    receptor           input = binding_row(ligand) * receptor_abundance(donor, cell_type)   (120-d)
    receptor_permuted  same, but the cell-type -> abundance map is permuted within each donor (null control)
    mlp_full_basal     input = [binding_row(ligand), full basal expression (2000 HVG) of PBS in (donor, cell_type)]
                       through a wider MLP (capacity control: can the gate find competence by itself?)

Both are plain arguments, so a papermill notebook takes them as parameters (``pert_rep``, ``gate``);
``add_arm_flags`` additionally exposes them as ``--pert_rep/--gate`` on any argparse CLI.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from collections.abc import Sequence

import numpy as np
import pandas as pd

PERT_REPS = ("constant", "esm2", "binding", "binding_esm2", "family_onehot")
GATES = ("none", "receptor", "receptor_permuted", "mlp_full_basal")

# Stage-C Tier-1 arms
ARMS = {
    "A0": ("constant", "none"),
    "A1": ("esm2", "none"),
    "A2": ("binding", "none"),
    "A3": ("binding", "receptor"),
}

GATE_KEY = "gate_input"
CONSTANT_DIM = 16


@dataclasses.dataclass(frozen=True)
class ArmConfig:
    pert_rep: str = "esm2"
    gate: str = "none"

    def __post_init__(self):
        if self.pert_rep not in PERT_REPS:
            raise ValueError(f"pert_rep must be one of {PERT_REPS}, got {self.pert_rep!r}")
        if self.gate not in GATES:
            raise ValueError(f"gate must be one of {GATES}, got {self.gate!r}")

    @classmethod
    def from_arm(cls, arm: str) -> ArmConfig:
        return cls(*ARMS[arm])


def add_arm_flags(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--pert_rep", choices=PERT_REPS, default="esm2")
    parser.add_argument("--gate", choices=GATES, default="none")
    return parser


def config_from_args(args: argparse.Namespace) -> ArmConfig:
    return ArmConfig(pert_rep=args.pert_rep, gate=args.gate)


# ------------------------------------------------------------------ features
def derange(n: int, seed: int) -> np.ndarray:
    """Permutation of range(n) without fixed points (n >= 2)."""
    rng = np.random.default_rng(seed)
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


def receptor_features(B: pd.DataFrame, R: pd.DataFrame, permute_seed: int | None = None) -> dict[str, np.ndarray]:
    """``{"donor|cell_type|ligand": B[ligand] * R[donor, cell_type]}`` over shared receptor subunits.

    ``B``: ligand x subunit (signed binding). ``R``: rows (donor, cell_type) + subunit columns (log1p CPM abundance).
    If ``permute_seed`` is given, the cell_type -> abundance assignment is permuted (derangement) within each donor.
    """
    subunits = [c for c in B.columns if c in R.columns]
    Bm = B[subunits]
    R = R.copy()
    if permute_seed is not None:
        parts = []
        for d, g in R.groupby("donor"):
            g = g.sort_values("cell_type").reset_index(drop=True)
            ab = g[subunits].to_numpy()[derange(len(g), permute_seed)]
            g[subunits] = ab
            parts.append(g)
        R = pd.concat(parts, ignore_index=True)
    out = {}
    for _, r in R.iterrows():
        a = r[subunits].to_numpy(dtype=np.float32)
        for lig, b in zip(Bm.index, Bm.to_numpy(dtype=np.float32)):
            out[f"{r['donor']}|{r['cell_type']}|{lig}"] = b * a
    return out


def basal_features(B: pd.DataFrame, basal: dict[str, np.ndarray], donors, cell_types, ligands) -> dict[str, np.ndarray]:
    """``mlp_full_basal`` input: [binding row (120), basal expression of (donor, cell_type)]."""
    out = {}
    for d in donors:
        for c in cell_types:
            k = f"{d}|{c}"
            if k not in basal:
                continue
            for lig in ligands:
                out[f"{d}|{c}|{lig}"] = np.concatenate([B.loc[lig].to_numpy(np.float32), basal[k].astype(np.float32)])
    return out


def pert_rep_vectors(cfg: ArmConfig, ligands: Sequence[str], esm2: dict, B: pd.DataFrame, family: dict[str, str]):
    ligands = list(ligands)
    if cfg.pert_rep == "constant":
        return {l: np.ones(CONSTANT_DIM, np.float32) for l in ligands}
    if cfg.pert_rep == "esm2":
        return {l: np.asarray(esm2[l], np.float32) for l in ligands}
    if cfg.pert_rep == "binding":
        return {l: B.loc[l].to_numpy(np.float32) for l in ligands}
    if cfg.pert_rep == "binding_esm2":
        return {l: np.concatenate([B.loc[l].to_numpy(np.float32), np.asarray(esm2[l], np.float32)]) for l in ligands}
    if cfg.pert_rep == "family_onehot":
        fams = sorted(set(family.values()))
        # ligands with no known family (absent from obs) get the zero vector instead of raising
        return {l: (np.eye(len(fams), dtype=np.float32)[fams.index(family[l])] if l in family else np.zeros(len(fams), np.float32)) for l in ligands}
    raise ValueError(cfg.pert_rep)


def prepare_arm(adata, cfg: ArmConfig, data_dir: str, seed: int = 0, basal_path: str | None = None):
    """Attach arm-specific reps to ``adata`` (in place) and return ``(prepare_data_kwargs, prepare_model_kwargs)``.

    ``adata.obs`` needs: donor, cell_type, cytokine ("PBS" = control), cytokine_family; ``uns['esm2_embeddings']``.
    Call on the training adata AND the validation/test adata (they must share ``uns``), ligands = all 90.
    """
    B = pd.read_parquet(os.path.join(data_dir, "binding_B.parquet"))
    obs = adata.obs
    ligands = sorted(set(obs.cytokine.astype(str)) - {"PBS"} | set(B.index))
    family = (
        obs.assign(c=obs.cytokine.astype(str), f=obs.cytokine_family.astype(str))
        .drop_duplicates("c")
        .set_index("c")["f"]
        .to_dict()
        if "cytokine_family" in obs
        else {}
    )
    reps = pert_rep_vectors(cfg, ligands, adata.uns["esm2_embeddings"], B, family)
    adata.uns["pert_rep"] = reps
    cts = sorted(obs.cell_type.astype(str).unique())
    adata.uns["celltype_onehot"] = {c: np.eye(len(cts), dtype=np.float32)[i] for i, c in enumerate(cts)}

    pd_kwargs = dict(
        sample_rep="X_pca",
        control_key="is_control",
        perturbation_covariates={"cytokine_treatment": ("cytokine",)},
        perturbation_covariate_reps={"cytokine_treatment": "pert_rep"},
        sample_covariates=["donor", "cell_type"],
        sample_covariate_reps={"donor": "donor_embeddings", "cell_type": "celltype_onehot"},
        split_covariates=["donor", "cell_type"],  # each (donor, cell_type) is its own condition
        max_combination_length=1,
        null_value=0.0,
    )
    layers_before_pool = {
        "cytokine_treatment": {"layer_type": "mlp", "dims": [1024, 1024], "dropout_rate": 0.5},
        "donor": {"layer_type": "mlp", "dims": [256, 256], "dropout_rate": 0.0},
        "cell_type": {"layer_type": "mlp", "dims": [64, 64], "dropout_rate": 0.0},
    }
    pm_kwargs = dict(layers_before_pool=layers_before_pool, gate=cfg.gate, gate_key=GATE_KEY, gate_dims=(32,))

    if cfg.gate != "none":
        donors = sorted(obs.donor.astype(str).unique())
        if cfg.gate in ("receptor", "receptor_permuted"):
            R = pd.read_parquet(os.path.join(data_dir, "receptor_abundance_R.parquet"))
            feats = receptor_features(B, R, permute_seed=(seed + 12345) if cfg.gate == "receptor_permuted" else None)
        else:
            z = np.load(basal_path or os.path.join(data_dir, "basal_means_train90.npz"), allow_pickle=True)
            basal = {k: z["X"][i] for i, k in enumerate(z["keys"])}
            feats = basal_features(B, basal, donors, cts, ligands)
            pm_kwargs["gate_dims"] = (256, 32)
        dim = len(next(iter(feats.values())))
        feats["PBS"] = np.zeros(dim, np.float32)
        adata.uns["gate_rep"] = feats
        obs["gate_id"] = np.where(
            obs.cytokine.astype(str) == "PBS", "PBS", obs.donor.astype(str) + "|" + obs.cell_type.astype(str) + "|" + obs.cytokine.astype(str)
        )
        pd_kwargs["perturbation_covariates"][GATE_KEY] = ("gate_id",)
        pd_kwargs["perturbation_covariate_reps"][GATE_KEY] = "gate_rep"
    return pd_kwargs, pm_kwargs


def describe(cfg: ArmConfig) -> str:
    return json.dumps(dataclasses.asdict(cfg))


if __name__ == "__main__":  # python -m cellflow.arms --pert_rep binding --gate receptor
    ap = add_arm_flags(argparse.ArgumentParser())
    print(describe(config_from_args(ap.parse_args())))
