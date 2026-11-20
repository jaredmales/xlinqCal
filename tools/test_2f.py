#!/usr/bin/env python3
"""Test the second-order picture: the even part of mode k at +-2f is predicted by the 2n pair's odd parts.

For m_k = sqrt(2) cos(psi + theta_k) and phase kappa*a_k*m_k, the second-order field -phi^2/2 has a
component -(kappa a_k)^2/4 e^{+-2i theta_k} A(xi -+ 2f) at +-2f. It interferes with C, so

    even_k(xi) = kappa * pred(xi),   pred = -(a_k^2 / (2 sqrt 2)) Re(e^{+-2i theta_k} Z_2n(xi)),

where Z_2n = sqrt(2) kappa C* A(xi -+ 2f) comes from the odd parts of the cos/sin pair at 2n (same
A shift). The only free parameter is kappa [rad of phase per um of DM command]. Per mode and side, the
even part in a window around +-2f is fit with pred, the reference (halo dimming), the mode's own
speckle (PSF shifted to the speckle), the |S2|^2 ghost speckle (PSF shifted to 2f, coefficient
kappa^4 a^4 / 16) and a constant.

Checks: kappa has the same sign on both sides (a negative kappa means --side-sign is the twin), and its
magnitude agrees with sqrt(E/a^2) from the speckle energy and with the ghost coefficient. Noise in the
2n odd parts biases kappa toward 0; the reliability factor (1 - noise var / pred var) corrects that.
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

    rows = []
    for key, ks in pairs.items():
        nn = np.array(key, float)
        if np.hypot(*nn) < args.nmin or np.abs(2 * nn).max() > nmax:
            continue
        q1, q2 = pairs[tuple((2 * nn).astype(int))]
        for k in ks:
            for s in (1, -1):
                p2 = c + s * args.side_sign * (M @ (2 * nn))
                w = np.hypot(xx - p2[0], yy - p2[1]) < args.radius
                if w.sum() < 10:
                    continue
                Z = xf.solve_Z(resp[q1][w] / amp[q1], resp[q2][w] / amp[q2], th[q1], th[q2], s)
                pred = -(amp[k] ** 2 / (2 * np.sqrt(2))) * np.real(np.exp(s * 2j * th[k]) * Z)
                p1 = s * args.side_sign * (M @ nn)
                own = map_coordinates(mref, [yy[w] - p1[1], xx[w] - p1[0]], order=1)
                ghost = map_coordinates(mref, [yy[w] - (p2[1] - c[1]), xx[w] - (p2[0] - c[0])], order=1)
                X = np.c_[pred, mref[w], own, ghost, np.ones(w.sum())]
                y = even[k][w].astype(np.float64)
                coef, *_ = np.linalg.lstsq(X, y, rcond=None)
                r = y - X @ coef
                cov = np.linalg.inv(X.T @ X) * r.var() * len(y) / (len(y) - X.shape[1])
                pnoise = (amp[k] ** 2 / (2 * np.sqrt(2))) ** 2 * 0.5 * ((sig[q1] / amp[q1]) ** 2
                                                                         + (sig[q2] / amp[q2]) ** 2)
                rows.append([s, np.hypot(*nn), coef[0], np.sqrt(cov[0, 0]), coef[3] / amp[k] ** 4,
                             np.sqrt(cov[3, 3]) / amp[k] ** 4, pnoise, np.var(pred)])
    rows = np.array(rows)
    side, fr, kap, sk, g, sg, pn, pv = rows.T

    km, ke = wmean(kap, sk)
    lam = 1 - pn.sum() / pv.sum()
    chi2 = np.sum((kap - km) ** 2 / sk ** 2) / (len(kap) - 1)
    print(f"{args.data}: {len(rows)} mode-sides, |n| {fr.min():.1f}-{fr.max():.1f}, side-sign {args.side_sign:+d}")
    print(f"  kappa {km:.2f} +- {ke:.2f} rad/um (chi2/dof {chi2:.2f}); reliability {lam:.2f}, "
          f"corrected {km / lam:.2f} +- {ke / lam:.2f}")
    for sv in (1, -1):
        m = side == sv
        a, b = wmean(kap[m], sk[m])
        print(f"  side {sv:+d}: kappa {a:.2f} +- {b:.2f}")
    for lo, hi in [(2.5, 4), (4, 6), (6, 8), (8, 12), (12, 18)]:
        m = (fr >= lo) & (fr < hi)
        if m.any():
            a, b = wmean(kap[m], sk[m])
            lm = 1 - pn[m].sum() / pv[m].sum()
            print(f"  |n| {lo:4.1f}-{hi:<4.1f}: N {m.sum():4d}  kappa {a:6.2f} +- {b:.2f}  reliability {lm:.2f}  "
                  f"corrected {a / lm:6.2f} +- {b / lm:.2f}")
    gm, ge = wmean(g, sg)
    kg = (16 * max(gm, 0)) ** 0.25
    print(f"  ghost |S2|^2: kappa^4/16 = {gm:.1f} +- {ge:.1f} -> kappa {kg:.2f} "
          f"(range {(16 * max(gm - ge, 0)) ** 0.25:.2f}-{(16 * (gm + ge)) ** 0.25:.2f})")
    if km < 0:
        print("  kappa < 0: the side sign is the twin; rerun with the other --side-sign")


if __name__ == "__main__":
    main()
