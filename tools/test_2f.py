#!/usr/bin/env python3
"""Test the second-order picture: the even part of mode k at +-2f is predicted by the 2n pair's odd parts.

For m_k = sqrt(2) cos(psi + theta_k) and phase kappa*a_k*m_k, the second-order field -phi^2/2 has a
component -(kappa a_k)^2/4 e^{+-2i theta_k} A(xi -+ 2f) at +-2f. It interferes with C, so

    even_k(xi) = kappa * pred(xi),   pred = -(a_k^2 / (2 sqrt 2)) Re(e^{+-2i theta_k} Z_2n(xi)),

where Z_2n = sqrt(2) kappa C* A(xi -+ 2f) comes from the odd parts of the cos/sin pair at 2n (same
A shift). The only free parameter is kappa [rad of phase per um of DM command]. Per mode and side, the
even part in a window around +-2f is fit with pred, the reference (halo dimming), the mode's own
speckle (PSF shifted to the speckle), the |S2|^2 ghost speckle (PSF shifted to 2f, coefficient
kappa^4 a^4 / 16), a constant, and a DM 2f harmonic.

DM nonlinearity (actuator coupling) adds surface gamma a^2 cos(2 psi + 2 theta_k) [um] at 2f, aligned with
cos^2 by symmetry. Its field is in quadrature with the optical term, so it gets its own regressor,
pred_dm = (a_k^2 / sqrt 2) Re(i e^{+-2i theta_k} Z_2n), with coefficient gamma [um^-1]. The ratio of 2f to
fundamental surface is gamma a / sqrt 2. The DM channel doesn't depend on --side-sign.

Within one window the optical and DM regressors are nearly collinear (arg Z varies little over a
speckle), so they are separated across windows: the nuisance terms are projected out per window and
(kappa, gamma) are fit jointly to all windows. Noise in the 2n odd parts biases the fit toward 0; the
known noise variance is subtracted from the normal matrix. Errors are bootstrap over windows.

Checks: kappa and gamma agree between the two sides (a negative kappa means --side-sign is the twin),
and kappa agrees with 4 pi / lambda for surface-um commands, with sqrt(E/a^2) from the speckle energy,
and with the ghost coefficient.
"""

import argparse

import numpy as np
from scipy.ndimage import map_coordinates

import xlq_fourier as xf


def wmean(v, s):
    w = 1 / s ** 2
    return np.sum(w * v) / w.sum(), 1 / np.sqrt(w.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", required=True, help="the mode cube (1 um rms modes)")
    ap.add_argument("--data", required=True, help="data set prefix, e.g. try2/xlinqCal_20261119_002")
    ap.add_argument("--sci", default="camsci1", help="science stream name in the output files")
    ap.add_argument("--grid", type=int, default=50, help="actuators across the mode grid")
    ap.add_argument("--side-sign", type=int, default=-1, choices=[-1, 1],
                    help="e^{+i psi} speckle at c + side_sign*M@n; -1 was measured on try1/try2")
    ap.add_argument("--nmin", type=float, default=2.5, help="minimum |n|")
    ap.add_argument("--radius", type=float, default=4.0, help="window radius around +-2f [px]")
    args = ap.parse_args()

    modes = xf.load_modes(args.modes)
    n, R, th, res, pairs = xf.mode_table(modes, args.grid)
    d = xf.load_set(args.data, modes, args.sci)
    M, gres, ngood = xf.geometry(d, n)
    resp, even, mref, amp, c, sig, yy, xx = (d[k] for k in ("resp", "even", "mref", "amp", "c", "sig", "yy", "xx"))
    nmax = (args.grid - 1) // 2

    win = []  # per window: side, |n|, a, ghost coef/a^4, its sigma, sufficient stats for the joint fit
    for key, ks in pairs.items():
        nn = np.array(key, float)
        if np.hypot(*nn) < args.nmin or np.abs(2 * nn).max() > nmax:
            continue
        q1, q2 = pairs[tuple((2 * nn).astype(int))]
        zb = 0.5 * ((sig[q1] / amp[q1]) ** 2 + (sig[q2] / amp[q2]) ** 2)  # noise var of Re or Im of Z
        for k in ks:
            for s in (1, -1):
                p2 = c + s * args.side_sign * (M @ (2 * nn))
                w = np.hypot(xx - p2[0], yy - p2[1]) < args.radius
                nw = int(w.sum())
                if nw < 10:
                    continue
                Z = xf.solve_Z(resp[q1][w] / amp[q1], resp[q2][w] / amp[q2], th[q1], th[q2], s)
                e2 = np.exp(s * 2j * th[k])
                pred = -(amp[k] ** 2 / (2 * np.sqrt(2))) * np.real(e2 * Z)
                pdm = (amp[k] ** 2 / np.sqrt(2)) * np.real(1j * e2 * Z)
                p1 = s * args.side_sign * (M @ nn)
                own = map_coordinates(mref, [yy[w] - p1[1], xx[w] - p1[0]], order=1)
                ghost = map_coordinates(mref, [yy[w] - (p2[1] - c[1]), xx[w] - (p2[0] - c[0])], order=1)
                y = even[k][w].astype(np.float64)

                # ghost coefficient from the full per-window fit
                X = np.c_[pred, pdm, mref[w], own, ghost, np.ones(nw)]
                coef, *_ = np.linalg.lstsq(X, y, rcond=None)
                r = y - X @ coef
                cov = np.linalg.pinv(X.T @ X) * r.var() * nw / (nw - X.shape[1])

                # nuisance-projected regressors for the joint (stacked) fit
                Nu = np.c_[mref[w], own, ghost, np.ones(nw)]
                proj = lambda v: v - Nu @ np.linalg.lstsq(Nu, v, rcond=None)[0]  # noqa: E731
                P = np.c_[proj(pred), proj(pdm)]
                yr = proj(y)
                vo = (amp[k] ** 2 / (2 * np.sqrt(2))) ** 2 * zb * (nw - Nu.shape[1])
                win.append(dict(side=s, fr=np.hypot(*nn), a=amp[k], g=coef[4] / amp[k] ** 4,
                                sg=np.sqrt(cov[4, 4]) / amp[k] ** 4, PP=P.T @ P, Py=P.T @ yr,
                                noise=np.diag([vo, 4 * vo])))

    side = np.array([d_["side"] for d_ in win])
    fr = np.array([d_["fr"] for d_ in win])
    ak = np.array([d_["a"] for d_ in win])
    PP = np.array([d_["PP"] for d_ in win])
    Py = np.array([d_["Py"] for d_ in win])
    NZ = np.array([d_["noise"] for d_ in win])
    rng = np.random.default_rng(1)

    def joint(idx, nboot=300):
        """kappa, gamma (noise-corrected), their bootstrap sigmas, and the uncorrected kappa."""
        def solve(ii, corr=True):
            A = PP[ii].sum(0) - (NZ[ii].sum(0) if corr else 0)
            return np.linalg.solve(A, Py[ii].sum(0))
        est = solve(idx)
        raw = solve(idx, False)
        if nboot == 0:
            return est, np.full(2, np.nan), raw
        bs = np.array([solve(rng.choice(idx, len(idx))) for _ in range(nboot)])
        return est, bs.std(0), raw

    print(f"{args.data}: {len(win)} mode-sides, |n| {fr.min():.1f}-{fr.max():.1f}, side-sign {args.side_sign:+d}")
    print("  joint stacked fit, noise-corrected, bootstrap errors (kappa: optical, rad/um; "
          "gamma: DM 2f surface, um^-1; 2f/fund = gamma a/sqrt2 at the median a)")
    groups = [("all", np.arange(len(win)))]
    groups += [(f"side {sv:+d}", np.where(side == sv)[0]) for sv in (1, -1)]
    groups += [(f"|n| {lo:4.1f}-{hi:<4.1f}", np.where((fr >= lo) & (fr < hi))[0])
               for lo, hi in [(2.5, 4), (4, 6), (6, 8), (8, 12), (12, 18)]]
    for name, idx in groups:
        if len(idx) < 10:
            continue
        (kap, gam), (sk, sgm), raw = joint(idx)
        print(f"  {name:14s} N {len(idx):4d}: kappa {kap:6.2f} +- {sk:.2f} (uncorrected {raw[0]:6.2f})   "
              f"gamma {gam:7.3f} +- {sgm:.3f}   2f/fund {gam * np.median(ak[idx]) / np.sqrt(2):+.4f}")

    g = np.array([d_["g"] for d_ in win])
    sg = np.array([d_["sg"] for d_ in win])
    gm, ge = wmean(g, sg)
    print(f"  ghost |S2|^2: kappa^4/16 = {gm:.1f} +- {ge:.1f} -> kappa {(16 * max(gm, 0)) ** 0.25:.2f} "
          f"(range {(16 * max(gm - ge, 0)) ** 0.25:.2f}-{(16 * max(gm + ge, 0)) ** 0.25:.2f})")
    if joint(np.arange(len(win)), 0)[0][0] < 0:
        print("  kappa < 0: the side sign is the twin; rerun with the other --side-sign")


if __name__ == "__main__":
    main()
