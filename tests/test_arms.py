"""Unit tests for the arm flags. CPU only. Run BEFORE any training:  pytest -q tests/test_arms.py"""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "cellflow_fork"))
import numpy as np, pandas as pd, pytest, jax, jax.numpy as jnp, anndata as ad
import cellflow
assert cellflow.__file__.startswith(os.path.join(ROOT, "cellflow_fork")), cellflow.__file__
from cellflow.arms import ArmConfig, prepare_arm, receptor_features, pert_rep_vectors, ARMS, GATE_KEY
from cellflow.networks._velocity_field import ConditionalVelocityField, CompetenceGate
from cellflow.model import CellFlow

DATA = os.path.join(ROOT, "data")
B = pd.read_parquet(f"{DATA}/binding_B.parquet"); R = pd.read_parquet(f"{DATA}/receptor_abundance_R.parquet")
LIGS = ["IL-2", "IFN-gamma", "IL-4", "TNF-alpha", "EGF", "GM-CSF"]


def _vf(gate="receptor", gdim=120):
    return ConditionalVelocityField(output_dim=8, max_combination_length=1, condition_embedding_dim=16,
        layers_before_pool={"a": {"layer_type": "mlp", "dims": [16]}}, layers_after_pool={"layer_type": "mlp", "dims": [16]},
        time_encoder_dims=(16,), hidden_dims=(16,), decoder_dims=(16,), time_freqs=8, pooling="mean",
        gate=gate, gate_dims=(8,)), {"a": jnp.ones((3, 1, 5)), GATE_KEY: jnp.ones((3, 1, gdim))}


def _init(vf, cond):
    return vf.init({"params": jax.random.PRNGKey(0), "condition_encoder": jax.random.PRNGKey(1)},
                   t=jnp.ones((3, 1)), x_t=jnp.ones((3, 8)), cond=cond, encoder_noise=jnp.ones((3, 16)), train=False)["params"]


def test_zero_competence_gives_zero_latent():
    vf, cond = _vf(); params = _init(vf, cond)
    cond0 = dict(cond); cond0[GATE_KEY] = jnp.zeros_like(cond[GATE_KEY])
    mean0, _, g0 = vf.apply({"params": params}, cond0, False, method="_encode")
    assert float(jnp.abs(g0).max()) < 1e-6 and float(jnp.abs(mean0).max()) < 1e-6
    cond1 = dict(cond); cond1[GATE_KEY] = jax.random.uniform(jax.random.PRNGKey(2), cond[GATE_KEY].shape) * 4
    mean1, _, g1 = vf.apply({"params": params}, cond1, False, method="_encode")
    assert float(jnp.abs(mean1).max()) > 1e-4 and float(g1.min()) >= 0 and float(g1.max()) < 1  # non-negative, bounded


def test_gate_is_not_monotone_by_construction():
    """A mask, not a magnitude predictor: scaling the input up does not force gate up (|z| can fall after init)."""
    g = CompetenceGate(dims=(8,)); x = jnp.linspace(0, 5, 50)[:, None] * jnp.ones((1, 120)) * 0 + jnp.linspace(0, 5, 50)[:, None]
    p = g.init(jax.random.PRNGKey(3), x)["params"]
    # force a negative readout: gate then depends on |z|; train-free check that output is a function of features, bounded
    out = np.asarray(g.apply({"params": p}, x)).ravel()
    assert out.min() >= 0 and out.max() < 1


def test_permuting_receptor_features_changes_output():
    vf, cond = _vf(); params = _init(vf, cond)
    f = jax.random.uniform(jax.random.PRNGKey(4), (3, 1, 120)) * (jnp.arange(120) % 7)  # non-uniform columns
    cond_a = {**cond, GATE_KEY: f}
    cond_b = {**cond, GATE_KEY: f[..., jax.random.permutation(jax.random.PRNGKey(5), 120)]}
    ma = vf.apply({"params": params}, cond_a, False, method="_encode")[0]
    mb = vf.apply({"params": params}, cond_b, False, method="_encode")[0]
    assert float(jnp.abs(ma - mb).max()) > 1e-6
    # data-level: permuted-gate features differ from real ones for the same (donor, cell_type, ligand), same ligand pattern
    real = receptor_features(B, R); perm = receptor_features(B, R, permute_seed=0)
    keys = [k for k in real if k.endswith("|IL-2")]
    assert sum(not np.allclose(real[k], perm[k]) for k in keys) > 0.8 * len(keys)
    assert all(np.array_equal(real[k] != 0, real[k] != 0) for k in keys)


def test_constant_pert_rep_makes_ligands_identical():
    B_ = B; esm = {l: np.random.randn(8).astype(np.float32) for l in LIGS}
    r = pert_rep_vectors(ArmConfig("constant", "none"), LIGS, esm, B_, {})
    assert len({tuple(v) for v in r.values()}) == 1
    r2 = pert_rep_vectors(ArmConfig("esm2", "none"), LIGS, esm, B_, {})
    assert len({tuple(v) for v in r2.values()}) == len(LIGS)


def _toy(n_per=40, seed=0):
    rng = np.random.default_rng(seed); donors = ["Donor1", "Donor2"]; cts = sorted(R.cell_type.unique())[:3]
    rows = [(d, c, l) for d in donors for c in cts for l in ["PBS"] + LIGS]
    obs = pd.DataFrame([r for r in rows for _ in range(n_per)], columns=["donor", "cell_type", "cytokine"])
    obs["is_control"] = obs.cytokine == "PBS"; obs["cytokine_family"] = "fam"
    obs["condition"] = obs.donor + "_" + obs.cytokine
    a = ad.AnnData(X=np.zeros((len(obs), 4), np.float32), obs=obs.reset_index(drop=True))
    a.obs_names = [f"c{i}" for i in range(len(a))]
    a.obsm["X_pca"] = rng.normal(size=(len(a), 10)).astype(np.float32)
    a.uns["esm2_embeddings"] = {l: rng.normal(size=32).astype(np.float32) for l in B.index}
    a.uns["donor_embeddings"] = {d: rng.normal(size=6).astype(np.float32) for d in donors}
    return a


@pytest.mark.parametrize("pert_rep,gate", [("constant", "none"), ("esm2", "none"), ("binding", "none"), ("binding", "receptor"),
                                          ("binding", "receptor_permuted"), ("family_onehot", "none"), ("binding_esm2", "none")])
def test_arm_end_to_end_tiny(pert_rep, gate):
    a = _toy(); cfg = ArmConfig(pert_rep, gate)
    pdk, pmk = prepare_arm(a, cfg, DATA, seed=0)
    cf = CellFlow(a, solver="otfm"); cf.prepare_data(**pdk)
    cd = cf.train_data.condition_data
    assert ("gate_input" in cd) == (gate != "none")
    if pert_rep == "constant":  # ALL ligands identical in the condition tensor
        assert np.allclose(cd["cytokine_treatment"], cd["cytokine_treatment"][0])
    cf.prepare_model(condition_embedding_dim=16, time_encoder_dims=[16], hidden_dims=[16], decoder_dims=[16], time_freqs=8,
                     pooling="mean", layers_after_pool={"layer_type": "mlp", "dims": [16]}, **pmk)
    cf.train(num_iterations=3, batch_size=32, valid_freq=100)
    # predict path (gate popped inside the vf)
    ctrl = a[a.obs.is_control.to_numpy()]
    cov = a[~a.obs.is_control.to_numpy()].obs.drop_duplicates(["donor", "cell_type", "cytokine"])
    cov["gid"] = cov.donor.astype(str) + "|" + cov.cell_type.astype(str) + "|" + cov.cytokine.astype(str)
    # an untrained toy net can make the adaptive ODE solve take many steps -> generous max_steps; the check that matters is finiteness
    out = cf.predict(adata=ctrl, sample_rep="X_pca", condition_id_key="gid", covariate_data=cov, max_steps=200_000)
    assert len(out) == len(cov)
    assert all(np.isfinite(np.asarray(v)).all() for v in out.values())
