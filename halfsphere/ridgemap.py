import math

import torch
from torch.func import jvp, vjp

from .geometry import exp_map, tangent


class DecoderOps:
    def __init__(self, bb, v, mA, alpha=0.0, c=None, jvp_mode="dvjp", fd_eps_deg=0.2):
        assert v.shape[0] == 1, "Ridge Map works per image"
        assert bb.compute_dtype == torch.float32, "Ridge Map needs fp32 networks (trap T3)"
        self.bb, self.v, self.mA, self.alpha, self.c = bb, v.detach().float(), mA, alpha, c
        self.jvp_mode, self.fd_eps = jvp_mode, math.radians(fd_eps_deg)
        self.fn = lambda u: bb.decode(u, alpha, c)
        with bb.differentiable_attention():
            self.x0, self._vjp = vjp(self.fn, self.v)

    def P(self, xi):
        return tangent(self.v, xi)

    def Jt(self, w):
        with self.bb.differentiable_attention():
            out = self._vjp(w)[0]
        self.bb.counters["dec_bwd"] += 1
        return self.P(out)

    def J(self, xi):
        xi = self.P(xi)
        with self.bb.differentiable_attention():
            return self._J(xi)

    def _J(self, xi):
        if self.jvp_mode == "fwd":
            return jvp(self.fn, (self.v,), (xi,))[1]
        if self.jvp_mode == "fd":
            n = xi.norm() / math.sqrt(self.bb.L)
            d = xi / n.clamp_min(1e-30)
            h = self.fd_eps
            plus = self.fn(exp_map(self.v, h * d))
            minus = self.fn(exp_map(self.v, -h * d))
            return (plus - minus) / (2 * h) * n
        self.bb.counters["dec_bwd"] += 2
        u0 = torch.zeros_like(self.x0)
        _, g_vjp = vjp(lambda u: self._vjp(u)[0], u0)
        return g_vjp(xi)[0]

    def JA(self, xi):
        return self.mA * self.J(xi)

    def JB(self, xi):
        return (1 - self.mA) * self.J(xi)

    def GA(self, xi):
        return self.Jt(self.mA * self.J(xi))

    def GB(self, xi):
        return self.Jt((1 - self.mA) * self.J(xi))


def _dot(a, b):
    return (a * b).sum()


def cg(op, b, iters=30, mu=0.0, tol=1e-6):
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs = _dot(r, r)
    for _ in range(iters):
        Ap = op(p) + mu * p
        a = rs / _dot(p, Ap).clamp_min(1e-30)
        x = x + a * p
        r = r - a * Ap
        rs_new = _dot(r, r)
        if rs_new.sqrt() < tol * b.norm():
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x


def lanczos(op, v0, k=64):
    Q, alphas, betas = [], [], []
    q = v0 / v0.norm()
    beta, q_prev = torch.tensor(0.0, device=v0.device), torch.zeros_like(v0)
    for _ in range(k):
        Q.append(q)
        w = op(q) - beta * q_prev
        a = _dot(w, q)
        alphas.append(a)
        w = w - a * q
        for qi in Q:
            w = w - _dot(w, qi) * qi
        for qi in Q:
            w = w - _dot(w, qi) * qi
        beta = w.norm()
        betas.append(beta)
        if beta < 1e-10:
            break
        q_prev, q = q, w / beta
    m = len(alphas)
    T = torch.diag(torch.stack(alphas))
    if m > 1:
        off = torch.stack(betas[: m - 1])
        T = T + torch.diag(off, 1) + torch.diag(off, -1)
    theta, S = torch.linalg.eigh(T.double())
    ritz = torch.einsum("ji,j...->i...", S, torch.stack(Q[:m]).double()).to(v0.dtype)
    return theta.to(v0.dtype), ritz, T


def lm_step(ops, y, mu=1e-2, iters=30, max_deg=3.0):
    r_A = ops.mA * (ops.x0 - y)
    xi = ops.P(cg(ops.GA, -ops.Jt(r_A), iters=iters, mu=mu))
    ang = xi.norm() / math.sqrt(ops.bb.L)
    cap = math.radians(max_deg)
    if ang > cap:
        xi = xi * (cap / ang)
    return exp_map(ops.v, xi)


def lm_refine(bb, v, y, mA, steps=10, mu=1e-2, alpha=0.0, c=None, jvp_mode="dvjp", max_deg=3.0, cg_iters=30):
    hist = []
    for _ in range(steps):
        ops = DecoderOps(bb, v, mA, alpha, c, jvp_mode)
        loss = ((mA * (ops.x0 - y)) ** 2).mean().item()
        hist.append(loss)
        v_new = lm_step(ops, y, mu, iters=cg_iters, max_deg=max_deg).detach()
        with torch.no_grad():
            new_loss = ((mA * (bb.decode(v_new, alpha, c) - y)) ** 2).mean().item()
        if new_loss < loss:
            v, mu = v_new, mu * 0.3
        else:
            mu *= 10.0
    with torch.no_grad():
        hist.append(((mA * (bb.decode(v, alpha, c) - y)) ** 2).mean().item())
    return v, hist


def slq_rank(ops, eps_rel=1e-3, probes=8, k=80, lam_max=None, generator=None):
    L = ops.bb.L
    eps_list = eps_rel if isinstance(eps_rel, (list, tuple)) else [eps_rel]

    def rnd():
        return torch.randn(ops.v.shape, generator=generator, device="cpu").to(ops.v.device)

    if lam_max is None:
        th, _, _ = lanczos(ops.GA, ops.P(rnd()), k=30)
        lam_max = th.max().item()
    runs = []
    for _ in range(probes):
        z = ops.P(torch.sign(rnd()))
        nz2 = _dot(z, z).item()
        th, _, T = lanczos(ops.GA, z, k=k)
        _, S = torch.linalg.eigh(T.double())
        runs.append((th.double(), (S[0] ** 2), nz2))
    out = {"lam_max": lam_max, "L": L}
    for eps in eps_list:
        est = torch.tensor([(w * (th > eps * lam_max).double()).sum().item() * nz2 for th, w, nz2 in runs])
        out[eps] = {
            "rank_eps": est.mean().item(),
            "rank_eps_sd": est.std().item() if len(est) > 1 else 0.0,
            "fiber_dim": (L - 1) - est.mean().item(),
        }
    return out if len(eps_list) > 1 else {**out, **out[eps_list[0]]}


def completion_modes(ops, gamma=10.0, k=80, n_modes=8, generator=None):
    def op(xi):
        Jx = ops.J(xi)
        mB = 1 - ops.mA
        return ops.Jt(mB * Jx) - gamma * ops.Jt(ops.mA * Jx)

    v0 = ops.P(torch.randn(ops.v.shape, generator=generator, device="cpu").to(ops.v.device))
    th, ritz, _ = lanczos(op, v0, k=k)
    order = torch.argsort(th, descending=True)[:n_modes]
    modes, th = ritz[order], th[order]
    leak, resid = [], []
    for m, t in zip(modes, th):
        Jm = ops.J(m)
        leak.append((ops.mA * Jm).norm() / ((1 - ops.mA) * Jm).norm().clamp_min(1e-30))
        resid.append((op(m) - t * m).norm() / (t.abs() * m.norm()).clamp_min(1e-30))
    return th, modes, torch.stack(leak), torch.stack(resid)


@torch.no_grad()
def render_mode(bb, v, mode, degrees=(-6, -3, 0, 3, 6), alpha=0.0, c=None):
    d = tangent(v, mode)
    d = d / d.flatten(1).norm(dim=1).view(-1, 1, 1) * math.sqrt(bb.L)
    return torch.cat([bb.decode(exp_map(v, math.radians(t) * d) if t != 0 else v, alpha, c) for t in degrees])


def contraction_radius(bb, x_star, y, mA, alpha, e_hat, iters=20, c=None, generator=None):
    mB = 1 - mA

    def G(xB):
        return mB * bb.project(mA * y + mB * xB, alpha, e_hat, c)

    x0 = (mB * x_star).detach()
    with bb.differentiable_attention():
        out0, g_vjp = vjp(G, x0)
        _, gg = vjp(lambda u: g_vjp(u)[0], torch.zeros_like(out0))
    d = mB * torch.randn(x_star.shape, generator=generator, device="cpu").to(x_star.device)
    d = d / d.norm()
    rates = []
    for _ in range(iters):
        with bb.differentiable_attention():
            Jd = gg(d)[0]
        n = Jd.norm()
        rates.append(n.item())
        d = Jd / n.clamp_min(1e-30)
    tail = rates[-5:]
    return sum(tail) / len(tail), rates


def cca(VA, VB, k_pca=512, reg=1e-3):
    def pca(X):
        X = X - X.mean(0)
        _, _, Vh = torch.linalg.svd(X, full_matrices=False)
        return X @ Vh[:k_pca].T

    A, Bm = pca(VA.flatten(1).double()), pca(VB.flatten(1).double())
    n = A.shape[0]
    Ia, Ib = torch.eye(A.shape[1], dtype=A.dtype), torch.eye(Bm.shape[1], dtype=A.dtype)
    Wa = torch.linalg.inv(torch.linalg.cholesky(A.T @ A / n + reg * Ia))
    Wb = torch.linalg.inv(torch.linalg.cholesky(Bm.T @ Bm / n + reg * Ib))
    return torch.linalg.svdvals(Wa @ (A.T @ Bm / n) @ Wb.T)


def cca_with_null(VA, VB, k_pca=512, n_perm=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    real = cca(VA, VB, k_pca)
    null = torch.stack([cca(VA, VB[torch.randperm(VB.shape[0], generator=g)], k_pca) for _ in range(n_perm)])
    return real, null.mean(0), null.std(0)
