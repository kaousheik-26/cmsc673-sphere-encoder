import math

import pytest
import torch
from torch.func import jacrev

from halfsphere.repaint import RepaintConfig, repaint_complete
from halfsphere.ridgemap import (
    DecoderOps,
    cca_with_null,
    completion_modes,
    contraction_radius,
    lm_refine,
    slq_rank,
)

H = 32


@pytest.mark.parametrize("mode", ["dvjp", "fwd", "fd"])
def test_jvp_modes_match_dense(setup, dense, mode):
    bb, _, mA, _ = setup
    v, Jfull, P = dense
    ops = DecoderOps(bb, v, mA[:1], jvp_mode=mode)
    xi = ops.P(torch.randn_like(v))
    ref = (Jfull @ (P @ xi.flatten())).view(1, 3, H, H)
    tol = 1e-3 if mode == "fd" else 1e-5
    assert torch.allclose(ops.J(xi), ref, atol=tol * ref.abs().max().item())


def test_vjp_matches_dense(setup, dense):
    bb, _, mA, _ = setup
    v, Jfull, P = dense
    ops = DecoderOps(bb, v, mA[:1])
    w = torch.randn(1, 3, H, H)
    ref = (P @ (Jfull.T @ w.flatten())).view_as(v)
    assert torch.allclose(ops.Jt(w), ref, atol=1e-5)


def test_slq_rank_close_to_dense(setup, dense):
    bb, _, mA, _ = setup
    v, Jfull, P = dense
    JA = mA[:1].expand(1, 3, H, H).flatten()[:, None] * Jfull
    G = P @ JA.T @ JA @ P
    ev = torch.linalg.eigvalsh(G.double())
    r_true = (ev > 1e-3 * ev.max()).sum().item()
    est = slq_rank(DecoderOps(bb, v, mA[:1]), eps_rel=1e-3, probes=6, k=60, generator=torch.Generator().manual_seed(0))
    assert abs(est["rank_eps"] - r_true) < 0.15 * bb.L, (est, r_true)


def test_completion_modes_match_dense_and_leak_little(setup, dense):
    bb, _, mA, _ = setup
    v, Jfull, P = dense
    m = mA[:1].expand(1, 3, H, H).flatten()
    JA, JB = m[:, None] * Jfull, (1 - m)[:, None] * Jfull
    ev = torch.linalg.eigvalsh((P @ (JB.T @ JB - 10.0 * JA.T @ JA) @ P).double())
    th, _, leak, resid = completion_modes(DecoderOps(bb, v, mA[:1]), gamma=10.0, k=80, n_modes=4, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(th.double(), ev[-4:].flip(0), rtol=1e-2), (th, ev[-4:])
    assert leak.max() < 0.2 and resid.max() < 0.05


def test_lm_refine_monotone(setup):
    bb, x, mA, y = setup
    v = bb.encode(x[:1] * mA[:1])
    _, hist = lm_refine(bb, v, y[:1], mA[:1], steps=5, mu=1e-1)
    assert hist[-1] <= hist[0]


def test_contraction_radius_matches_dense(setup):
    bb, _, mA, y = setup
    g = torch.Generator().manual_seed(1)
    e_hat = bb.random_latent(1, "cpu", g)
    xs = repaint_complete(bb, y[:1], mA[:1], RepaintConfig(K=6, alpha_start=40, alpha_end=40, final_clean=False))
    rho, _ = contraction_radius(bb, xs, y[:1], mA[:1], 40.0, e_hat, iters=40, generator=g)
    mB = 1 - mA[:1]

    def G(xb):
        return (mB * bb.project(mA[:1] * y[:1] + mB * xb, 40.0, e_hat)).flatten()

    Jg = jacrev(G)((mB * xs).detach()).reshape(3 * H * H, 3 * H * H)
    rho_true = torch.linalg.eigvals(Jg.double()).abs().max().item()
    assert abs(rho - rho_true) < 0.15 * max(rho_true, 1e-3), (rho, rho_true)


def test_cca_null_matches_wachter():
    g = torch.Generator().manual_seed(0)
    VA, VB = torch.randn(2000, 300, generator=g), torch.randn(2000, 300, generator=g)
    real, _, _ = cca_with_null(VA, VB, k_pca=100, n_perm=2)
    assert abs(real[0].item() - 2 * math.sqrt(0.05 * 0.95)) < 0.05
