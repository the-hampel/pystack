#!/usr/bin/env python
"""
Cast a self-energy Sigma(i nu_n) from one temperature to another.

Two methods, both usable for heating (beta_new < beta_old) and cooling (beta_new > beta_old):

  cast_snap(...)  nearest Matsubara *index* snapping. This is what `sigma_old(MatsubaraFreq(n, ...))`
                  gives you: TRIQS does NOT interpolate on a Matsubara mesh -- evaluate(imfreq, f, iw)
                  is literally `f(iw.n)`, an exact integer-index lookup. Cheap, but the error is
                  decided by how the two grids happen to line up (see grid_alignment).

  cast_dlr(...)   fit a DLR pole model to Sigma - Sigma_inf at beta_old, then re-evaluate the same
                  real-axis poles at the new Matsubara frequencies. Accuracy is set by `eps` and is
                  independent of grid alignment.

Both take/return a TRIQS Gf on a MeshImFreq with a matrix (or scalar-shaped) target. For a BlockGf,
loop over blocks.

Companion notebook with the full analysis: sigma_temperature_interpolation.ipynb

Usage
-----
    python sigma_temperature_cast.py                    # cooling  beta 20 -> 40  (SCALE = 2)
    python sigma_temperature_cast.py --scale 1/3        # heating  beta 20 -> 6.67 (exact, odd factor)
    python sigma_temperature_cast.py --scale 0.5        # heating  beta 20 -> 10   (worst alignment)
    python sigma_temperature_cast.py --model mott --scale 4
"""

from __future__ import annotations

import argparse
from fractions import Fraction

import numpy as np
from triqs.gfs import (Gf, MeshImFreq, MatsubaraFreq, make_gf_dlr, make_gf_dlr_imfreq,
                       find_w_max, fit_hermitian_tail)

__all__ = ["cast_snap", "cast_dlr", "dlr_pole_model", "eval_poles",
           "grid_alignment", "quasiparticle_weight", "nu_of"]


# =====================================================================================
#  helpers
# =====================================================================================

def nu_of(g):
    """Matsubara frequencies nu_n (real numbers) of a MeshImFreq Gf, in mesh/data order."""
    return np.array([complex(iw.value).imag for iw in g.mesh])


def eval_poles(w_l, f_l, z):
    """Evaluate a pole model  f(z) = sum_l f_l / (z - w_l)  at arbitrary complex z.

    Parameters
    ----------
    w_l : (r,) real array        pole positions on the real axis
    f_l : (r, ...) array         pole weights, any target shape
    z   : (N,) complex array     evaluation points

    Returns
    -------
    (N, ...) array
    """
    z = np.atleast_1d(np.asarray(z, dtype=complex))
    return np.einsum("nl,l...->n...", 1.0 / (z[:, None] - w_l[None, :]), f_l)


def dlr_pole_model(g_dyn, w_max, eps):
    """DLR-fit a *decaying* Gf (Sigma - Sigma_inf) and return it as a real-axis pole model.

    Returns
    -------
    w_l : (r,) array   PHYSICAL pole positions.  NOTE mesh.dlr_freq returns the dimensionless
                       omega_l in [-Lambda, Lambda] with Lambda = w_max * beta, so we divide by beta.
    f_l : (r, ...)     DLR coefficients = pole weights, such that
                       g(z) = sum_l f_l / (z - w_l).
                       (The MeshDLR docstring writes 1/(iw + omega_l); with the omega_l that
                       dlr_freq actually returns the correct sign is 1/(z - omega_l).)
    """
    g_c = make_gf_dlr(make_gf_dlr_imfreq(g_dyn, w_max, eps))
    return np.array(g_c.mesh.dlr_freq) / g_c.mesh.beta, g_c.data


def quasiparticle_weight(g, orb=(0, 0)):
    """Z = [1 - d ImSigma/d nu]^-1 from the two lowest positive Matsubara points.

    Beware: for a Sigma produced by cast_snap() while *cooling*, this returns exactly 1.0 by
    construction (the two lowest new frequencies snap to the same old index). See grid_alignment().
    """
    nu = nu_of(g)
    pos = nu > 0
    nu2 = nu[pos][:2]
    im2 = g.data[pos][:2][(slice(None),) + tuple(orb)].imag
    return 1.0 / (1.0 - (im2[1] - im2[0]) / (nu2[1] - nu2[0]))


# =====================================================================================
#  diagnostics -- worth running before trusting cast_snap
# =====================================================================================

def grid_alignment(beta_old, beta_new, n_iw_new, n_iw_old=None, n_report=6):
    """How do the two Matsubara grids line up? This decides the cast_snap error.

    The snapping error is ~ |Sigma'(i nu)| * offset, where offset <= pi/beta_old is the distance
    from each new frequency to the nearest old one. It does NOT depend on the size of the
    temperature jump, and adding more Matsubara points does not reduce it (the spacing is
    2*pi/beta_old; n_iw only extends the range).

    For heating by an integer factor r = beta_old/beta_new, parity decides everything:
        nu_n^new = (2n+1) * r * pi/beta_old  vs  nu_m^old = (2m+1) * pi/beta_old
        r odd  -> (2n+1)r odd  -> every new frequency COINCIDES with an old one (offset 0, exact)
        r even -> (2n+1)r even -> every new frequency sits exactly MIDWAY (offset 1, worst case)

    Returns a dict of diagnostics; also human-readable via print_alignment().
    """
    half = np.pi / beta_old                       # half the old spacing = max possible offset
    n = np.arange(n_iw_new)
    nu_new = (2 * n + 1) * np.pi / beta_new
    n_old = np.round(((2 * n + 1) * beta_old / beta_new - 1) / 2).astype(int)
    nu_old = (2 * n_old + 1) * np.pi / beta_old
    offset = np.abs(nu_old - nu_new) / half

    ratio = beta_old / beta_new
    collapses = bool(n_old[0] == n_old[1]) if n_iw_new > 1 else False
    n_dup = int(n_iw_new - len(set(n_old.tolist())))

    kind = "cooling" if beta_new > beta_old else ("heating" if beta_new < beta_old else "same beta")
    parity = None
    if beta_new < beta_old and abs(ratio - round(ratio)) < 1e-9:
        parity = "odd" if round(ratio) % 2 == 1 else "even"

    out = dict(kind=kind, ratio=ratio, parity=parity, offset=offset,
               offset_n0=float(offset[0]), offset_max=float(offset.max()),
               n_old=n_old, nu_new=nu_new, nu_old=nu_old,
               collapses=collapses, n_duplicate=n_dup, n_report=n_report,
               below_old_nu0=int((nu_new < np.pi / beta_old).sum()))
    if n_iw_old is not None:
        need = int(np.abs(n_old).max()) + 1
        out.update(n_iw_old=n_iw_old, n_iw_old_needed=need,
                   n_outside_window=int((np.abs(n_old) > n_iw_old - 1).sum()))
    return out


def print_alignment(a):
    """Pretty-print the dict returned by grid_alignment()."""
    print(f"  direction            : {a['kind']}  (beta_old/beta_new = {a['ratio']:.4f})")
    if a["parity"] == "odd":
        print("  integer heating factor: ODD  -> grids coincide exactly, snapping is an exact copy")
    elif a["parity"] == "even":
        print("  integer heating factor: EVEN -> every point exactly midway, WORST case for snapping")
    print(f"  offset (n=0 / max)   : {a['offset_n0']:.3f} / {a['offset_max']:.3f}"
          f"   [units of pi/beta_old; 0 = exact, 1 = worst]")
    if a["collapses"]:
        print(f"  !! the 2 lowest new frequencies snap to the SAME old index (n_old="
              f"{a['n_old'][0]}) -> staircase, and Z_snap == 1 exactly")
    if a["n_duplicate"]:
        print(f"  duplicated old values: {a['n_duplicate']} of {len(a['n_old'])} new points reuse "
              f"an already-used old index")
    if a["below_old_nu0"]:
        print(f"  new frequencies below pi/beta_old: {a['below_old_nu0']}  "
              f"(genuine extrapolation; no beta_old data constrains this region)")
    if "n_iw_old" in a:
        if a["n_outside_window"]:
            print(f"  !! {a['n_outside_window']} of {len(a['n_old'])} new points need |n_old| up to "
                  f"{a['n_iw_old_needed']} but the old mesh only has n_iw={a['n_iw_old']}")
            print("     -> those are SILENTLY replaced by the fitted high-frequency tail, no warning")
        else:
            print(f"  window coverage      : OK (needs n_iw_old >= {a['n_iw_old_needed']}, "
                  f"have {a['n_iw_old']})")
    print(f"  n_new=0..{a['n_report']-1} -> n_old = {a['n_old'][:a['n_report']].tolist()}")


# =====================================================================================
#  method A -- nearest Matsubara index ("plain Gf" snapping)
# =====================================================================================

def cast_snap(sigma_old, beta_new, n_iw_new, strict=True):
    """Cast Sigma to beta_new by snapping to the nearest Matsubara *index*.

    For each new frequency i nu_n^new this looks up Sigma at the old index
        n_old = round( ((2n+1) * beta_old/beta_new - 1) / 2 )
    Rounding is not a convenience: a MeshImFreq Gf cannot be evaluated at a non-integer index.

    Parameters
    ----------
    strict : bool
        If True (default) raise when the required old indices fall outside the old mesh window.
        TRIQS would otherwise silently substitute the fitted high-frequency tail. Set False to
        accept the tail (fine for Sigma - Sigma_inf ~ 1/i nu, but you should know it happened).

    Notes
    -----
    The `beta` argument of MatsubaraFreq is IGNORED by TRIQS -- only `.n` is used, together with
    the mesh's own beta. It is passed here only for readability.
    """
    beta_old = sigma_old.mesh.beta
    stat = sigma_old.mesh.statistic
    mesh_new = MeshImFreq(beta=beta_new, statistic=stat, n_iw=n_iw_new)

    if strict:
        a = grid_alignment(beta_old, beta_new, n_iw_new, n_iw_old=sigma_old.mesh.n_iw)
        if a["n_outside_window"]:
            raise ValueError(
                f"cast_snap: {a['n_outside_window']} new frequencies need old indices up to "
                f"{a['n_iw_old_needed']}, but the old mesh has n_iw={a['n_iw_old']}. Those values "
                f"would be silently taken from the fitted tail. Increase n_iw_old to "
                f">= {a['n_iw_old_needed']}, lower n_iw_new, or pass strict=False.")

    out = Gf(mesh=mesh_new, target_shape=sigma_old.target_shape)
    for iwn in mesh_new:
        n_old = round(((2 * iwn.index + 1) * beta_old / beta_new - 1) / 2)
        out[iwn] = sigma_old(MatsubaraFreq(n_old, beta_old, stat))
    return out


# =====================================================================================
#  method B -- DLR pole model
# =====================================================================================

def cast_dlr(sigma_old, beta_new, n_iw_new, eps=1e-10, w_max=None, sigma_inf=None,
             drop_below=None, return_info=False):
    """Cast Sigma to beta_new through a DLR pole model. Works for heating and cooling.

    Sigma_infinity is subtracted first and added back at the end: a constant is NOT in the span of
    {1/(z - w_l)} (every DLR basis function decays as 1/z), so an unsubtracted Sigma cannot be
    DLR-fitted at any w_max -- find_w_max will fail outright.

    Parameters
    ----------
    eps : float
        DLR accuracy. Must be >~ the noise level of `sigma_old`. Setting it far below the noise is
        actively harmful, not merely wasteful.
    w_max : float, optional
        DLR energy cutoff. Default: found automatically with find_w_max(). Note w_max is a PHYSICAL
        frequency and is held fixed across the temperature change (Lambda = w_max*beta is what
        changes) -- the poles are pinned to the spectrum, not to beta.
    sigma_inf : array (n,n), optional
        Static part. Default: fit_hermitian_tail(). STRONGLY prefer passing this explicitly (e.g.
        the Hartree term): on noisy data the tail fit amplifies the noise by ~1e2-1e3, and the
        leftover constant then makes find_w_max fail at every eps.
    drop_below : float, optional
        Discard poles with |w_l| < drop_below and refit the remaining weights by least squares over
        all Matsubara points. Passing pi/beta_old removes the poles the beta_old data cannot resolve
        and markedly improves *cooling*.
        WARNING: this asserts there is no real spectral weight below pi/beta_old, i.e. a Fermi
        liquid. For a Mott-like Sigma, where that weight is physical, it destroys the fit. Check
        that the fit residual on the original mesh does not degrade (returned as `fit_residual`).

    Cooling caveat
    --------------
    When cooling, the lowest new frequencies lie below pi/beta_old, which the beta_old data does not
    resolve. The DLR grid contains poles orders of magnitude closer to omega=0 than pi/beta_old; the
    fit gives them large, nearly cancelling weights that cancel for nu >~ pi/beta_old and stop
    cancelling below it. The result: the error is concentrated almost entirely in the n=0 point, it
    does NOT shrink with eps, and its size varies from one target component to another (an
    off-diagonal element can be far worse than the diagonals). `low_freq_amplification` in the info
    dict flags this -- if it is >> 1, distrust the lowest one or two cooled frequencies.

    Returns
    -------
    Gf  (and a dict of diagnostics if return_info=True)
    """
    beta_old = sigma_old.mesh.beta
    stat = sigma_old.mesh.statistic

    if sigma_inf is None:
        sigma_inf = fit_hermitian_tail(sigma_old)[0][0]
    sigma_inf = np.asarray(sigma_inf)

    dyn = sigma_old.copy()
    dyn.data[:] -= sigma_inf[None, ...]

    if w_max is None:
        w_max = find_w_max(dyn, eps)

    w_l, f_l = dlr_pole_model(dyn, w_max, eps)

    if drop_below is not None:
        keep = np.abs(w_l) >= drop_below
        w_l = w_l[keep]
        nu_old = nu_of(dyn)
        K = 1.0 / (1j * nu_old[:, None] - w_l[None, :])
        tgt = dyn.data.shape[1:]
        f_l = np.linalg.lstsq(K, dyn.data.reshape(len(nu_old), -1), rcond=None)[0]
        f_l = f_l.reshape((len(w_l),) + tgt)

    # how well does the model reproduce the input on its own mesh?
    fit_residual = float(np.abs(eval_poles(w_l, f_l, 1j * nu_of(dyn)) - dyn.data).max())

    out = Gf(mesh=MeshImFreq(beta=beta_new, statistic=stat, n_iw=n_iw_new),
             target_shape=sigma_old.target_shape)
    out.data[:] = eval_poles(w_l, f_l, 1j * nu_of(out)) + sigma_inf[None, ...]

    if return_info:
        # conditioning indicator: weight sitting on poles the beta_old data cannot resolve,
        # relative to the size of Sigma itself. >> 1 means the lowest cooled frequencies are unsafe.
        near0 = np.abs(w_l) < np.pi / beta_old
        scale = np.abs(dyn.data).max()
        amp = float(np.abs(f_l[near0]).sum(axis=0).max() / scale) if near0.any() and scale else 0.0
        return out, dict(w_max=w_max, eps=eps, rank=len(w_l), sigma_inf=sigma_inf,
                         w_l=w_l, f_l=f_l, fit_residual=fit_residual,
                         n_poles_below_nu0=int(near0.sum()),
                         low_freq_amplification=amp)
    return out


# =====================================================================================
#  synthetic test data (only needed to have an EXACT reference to compare against;
#  delete this block when using the script on real data)
# =====================================================================================

SIGMA_INF_REF = 3.5
_W = np.linspace(-15, 15, 1201)
_gauss = lambda w, w0, s: np.exp(-0.5 * ((w - w0) / s) ** 2) / (np.sqrt(2 * np.pi) * s)
_nrm = lambda a: a / a.sum()

REF_MODELS = {
    # A_Sigma(0) ~ 0 -> Fermi liquid, ImSigma(i nu) ~ -c*nu
    "fermi_liquid": _nrm(0.55 * _gauss(_W, -2.0, 0.40) + 0.45 * _gauss(_W, 2.4, 0.45)),
    # A_Sigma(0) > 0 -> finite scattering rate
    "bad_metal": _nrm(0.55 * _gauss(_W, -1.5, 0.60) + 0.45 * _gauss(_W, 1.9, 0.70)),
    # weight piled up at low energy -> Sigma nearly diverging as 1/i nu
    "mott": _nrm(0.50 * _gauss(_W, -0.15, 0.05) + 0.50 * _gauss(_W, 0.15, 0.05)),
}


def sigma_reference(z, model="fermi_liquid"):
    """Exact Sigma(z) from a fixed real-axis spectral density -- valid for ANY z, hence any beta."""
    z = np.atleast_1d(np.asarray(z, dtype=complex))
    return SIGMA_INF_REF + np.sum(REF_MODELS[model][None, :] / (z[:, None] - _W[None, :]), axis=1)


def make_reference_sigma(beta, n_iw, model="fermi_liquid"):
    """Sigma on a MeshImFreq, exact to machine precision (1x1 target)."""
    m = MeshImFreq(beta=beta, statistic="Fermion", n_iw=n_iw)
    g = Gf(mesh=m, target_shape=[1, 1])
    g.data[:, 0, 0] = sigma_reference([complex(iw.value) for iw in m], model)
    return g


# =====================================================================================
#  example
# =====================================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--beta-old", type=float, default=20.0)
    p.add_argument("--scale", default="2",
                   help="beta_new = beta_old * SCALE. Accepts fractions, e.g. '2' (cooling), "
                        "'1/3' (heating by the ODD factor 3 -> grids coincide exactly), "
                        "'1/2' (heating by an EVEN factor -> worst alignment). Default: 2")
    p.add_argument("--n-iw-old", type=int, default=200)
    p.add_argument("--model", default="fermi_liquid", choices=sorted(REF_MODELS))
    p.add_argument("--eps", type=float, default=1e-10)
    p.add_argument("--drop-below-nu0", action="store_true",
                   help="drop DLR poles below pi/beta_old (helps cooling; assumes a Fermi liquid)")
    p.add_argument("--out", default="sigma_cast.png")
    args = p.parse_args()

    scale = float(Fraction(args.scale))
    beta_old, beta_new = args.beta_old, args.beta_old * scale
    n_iw_old = args.n_iw_old
    n_iw_new = max(4, int(round(n_iw_old * scale)))       # keep nu_max roughly matched
    nu0_old = np.pi / beta_old

    print(f"\ncasting Sigma:  beta {beta_old:g} -> {beta_new:g}   (scale {args.scale} = {scale:.4f})")
    print(f"model '{args.model}',  n_iw {n_iw_old} -> {n_iw_new},  eps {args.eps:g}\n")

    sigma_old = make_reference_sigma(beta_old, n_iw_old, args.model)
    sigma_ref = make_reference_sigma(beta_new, n_iw_new, args.model)   # exact, for the error only

    print("grid alignment")
    print("--------------")
    align = grid_alignment(beta_old, beta_new, n_iw_new, n_iw_old=n_iw_old)
    print_alignment(align)

    # ---- the two casts
    sigma_snap = cast_snap(sigma_old, beta_new, n_iw_new, strict=False)
    sigma_dlr, info = cast_dlr(sigma_old, beta_new, n_iw_new, eps=args.eps,
                               sigma_inf=SIGMA_INF_REF * np.eye(1),
                               drop_below=nu0_old if args.drop_below_nu0 else None,
                               return_info=True)

    print("\nDLR model")
    print("---------")
    print(f"  w_max {info['w_max']:.4f} (Lambda = {info['w_max']*beta_old:.1f}),  rank {info['rank']}"
          f",  poles below pi/beta_old: {info['n_poles_below_nu0']}")
    print(f"  fit residual on the beta_old mesh: {info['fit_residual']:.2e}   (eps = {args.eps:g})")
    print(f"  low-frequency amplification: {info['low_freq_amplification']:.1f}x"
          f"   (weight on unresolvable poles / |Sigma|)")
    if beta_new > beta_old and info["low_freq_amplification"] > 3:
        print("  -> >> 1 while COOLING: expect the error to sit almost entirely in the n=0 point,")
        print("     and not to improve if you lower eps. Consider --drop-below-nu0 (Fermi liquid),")
        print("     or take smaller annealing steps with a re-converged Sigma at each stage.")
    if args.drop_below_nu0:
        print("  NOTE poles below pi/beta_old dropped -- assumes NO spectral weight there (Fermi")
        print("       liquid). If the residual above degraded badly, that assumption is wrong.")

    # ---- errors
    nu = nu_of(sigma_ref)
    pos = nu > 0
    e_snap = np.abs(sigma_snap.data[pos, 0, 0] - sigma_ref.data[pos, 0, 0])
    e_dlr = np.abs(sigma_dlr.data[pos, 0, 0] - sigma_ref.data[pos, 0, 0])

    print("\naccuracy vs the exact Sigma at beta_new")
    print("--------------------------------------")
    print(f"{'':<14}{'max|err|':>11}{'err at n=0':>13}{'err at n=1':>13}")
    for lbl, e in (("nearest index", e_snap), ("DLR", e_dlr)):
        print(f"{lbl:<14}{e.max():>11.2e}{e[0]:>13.2e}{e[1]:>13.2e}")

    print(f"\nquasiparticle weight Z (two lowest Matsubara points)")
    print("----------------------------------------------------")
    z_ref, z_snap, z_dlr = (quasiparticle_weight(g) for g in (sigma_ref, sigma_snap, sigma_dlr))
    print(f"  exact {z_ref:.5f}    nearest index {z_snap:.5f}    DLR {z_dlr:.5f}")
    if align["collapses"]:
        print("  (Z_snap = 1 exactly -- the staircase artifact, not physics)")

    # ---- plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                         "legend.frameon": False})

    fig, ax = plt.subplots(2, 3, figsize=(13.5, 6.2))
    curves = (("exact", sigma_ref, dict(color="k", ls="-", lw=1.6, marker=".", ms=5)),
              ("nearest index", sigma_snap, dict(color="tab:red", ls="--", lw=1.2, marker="s", ms=4)),
              ("DLR", sigma_dlr, dict(color="tab:blue", ls=":", lw=1.2, marker="o", ms=4)))
    zoom = pos & (nu < 10 * nu0_old)
    lbl_y = (r"${\rm Im}\,\Sigma$", r"${\rm Re}\,\Sigma$")

    for col, part in enumerate((np.imag, np.real)):
        a = ax[0, col]                                  # full range, log frequency axis
        for lbl, g, st in curves:
            a.plot(nu[pos], part(g.data[pos, 0, 0]), label=lbl, **st)
        a.set_xscale("log")
        a.set(xlabel=r"$\nu_n$", ylabel=lbl_y[col], title="full range")
        a.legend(fontsize=7)

        a = ax[1, col]                                  # low-frequency zoom, linear
        for lbl, g, st in curves:
            a.plot(nu[zoom], part(g.data[zoom, 0, 0]), label=lbl, **st)
        a.axvline(nu0_old, color="gray", ls=":", lw=1)
        a.annotate(r"$\pi/\beta_{\rm old}$", xy=(nu0_old, a.get_ylim()[0]), fontsize=7,
                   color="gray", va="bottom", ha="left")
        a.set(xlabel=r"$\nu_n$", ylabel=lbl_y[col], title="low-frequency zoom")
        a.legend(fontsize=7)

    for row, (xmax, ttl) in enumerate(((None, "absolute error"), (10 * nu0_old, "error, zoom"))):
        a = ax[row, 2]
        a.semilogy(nu[pos], e_snap + 1e-18, "s-", ms=3, color="tab:red", label="nearest index")
        a.semilogy(nu[pos], e_dlr + 1e-18, "o-", ms=3, color="tab:blue", label="DLR")
        a.axvline(nu0_old, color="gray", ls=":", lw=1)
        a.set(xlabel=r"$\nu_n$", ylabel=r"$|\Delta\Sigma|$", title=ttl,
              xlim=(0, xmax if xmax else min(12, nu[pos].max())))
        a.legend(fontsize=7)

    ttl = (f"$\\beta$ {beta_old:g} $\\to$ {beta_new:g}  ({align['kind']}, "
           f"$\\beta_{{\\rm old}}/\\beta_{{\\rm new}}$ = {align['ratio']:.3f}"
           + (f", integer {align['parity']} factor" if align["parity"] else "")
           + f") -- offset {align['offset_n0']:.2f}$\\,\\pi/\\beta_{{\\rm old}}$ at $n=0$")
    fig.suptitle(ttl, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out, dpi=120)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
