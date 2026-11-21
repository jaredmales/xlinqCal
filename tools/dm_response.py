#!/usr/bin/env python3
"""Measure the DM-to-phase response G(n) = kappa g(n) of each Fourier mode from its camsci1 speckle energy.

For a mode m = sqrt(2) cos(psi + theta) commanded at a [um], the phase is x sqrt(2) cos(.) with
x = G a, and the speckle pair holds 2 J1(sqrt(2) G a)^2 of the PSF flux, which is (G a)^2 for small
phase. Each pair's even-part energy is measured in apertures elongated radially by +-dl, because the
filter bandwidth smears speckles radially; the result is normalized by the PSF energy in a round
aperture of the same radius, which assumes the speckle has the PSF's shape. G is then solved with
the J1 form, so data sets taken at different amplitudes can be compared directly.

g(n) = G / kappa is the DM's response to the commanded surface. kappa = 4 pi / lambda for surface-um
commands (13.8 rad/um at 0.91 um). The 2f test (tools/test_2f.py) checks it independently, because
its fitted kappa is G(n)^2 / G(2n).

Prints G binned by |n| and a fit of kappa0 exp(-(f/f0)^p), where f = |n| / grid is in cycles per
actuator. Writes a FITS table with one row per mode: n_x, n_y, |n|, a, E, G, and g = G/kappa.
Not for focal-plane-mask data: it needs both speckles and an unblocked PSF.
"""

import argparse

import numpy as np
from astropy.io import fits
from scipy.optimize import brentq, least_squares
from scipy.special import j1

import xlq_fourier as xf


def capsule(xx, yy, c, q, w, dl):
    """Pixels within w px of the segment from c + q (1 - dl) to c + q (1 + dl)."""
    p0, p1 = c + q * (1 - dl), c + q * (1 + dl)
    v = p1 - p0
    L2 = max((v ** 2).sum(), 1e-12)
    t = np.clip(((xx - p0[0]) * v[0] + (yy - p0[1]) * v[1]) / L2, 0, 1)
    return np.hypot(xx - (p0[0] + t * v[0]), yy - (p0[1] + t * v[1])) < w


def solve_G(E, a):
    """G from the pair energy E (fraction of the PSF flux) at amplitude a, with the J1 form."""
    if not np.isfinite(E) or E <= 0:
        return np.nan
    xmax = 1.8  # J1 peaks at x = 1.84; beyond that the inversion is ambiguous
    f = lambda G: 2 * j1(np.sqrt(2) * G * a) ** 2 - E  # noqa: E731
    gmax = xmax / (np.sqrt(2) * a)
    return brentq(f, 1e-9, gmax) if f(gmax) > 0 else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", required=True, help="the mode cube (1 um rms modes)")
    ap.add_argument("--data", required=True, help="data set prefix, e.g. try2/xlinqCal_20261119_002")
    ap.add_argument("--sci", default="camsci1", help="science stream name in the output files")
    ap.add_argument("--grid", type=int, default=50, help="actuators across the mode grid")
    ap.add_argument("--radius", type=float, default=5.0, help="aperture radius [px]")
    ap.add_argument("--dl", type=float, default=0.08, help="radial half-elongation of the speckle aperture")
    ap.add_argument("--kappa", type=float, default=4 * np.pi / 0.91, help="4 pi / lambda [rad/um]")
    ap.add_argument("--out", help="output FITS table")
    args = ap.parse_args()

    modes = xf.load_modes(args.modes)
    n, R, th, res, pairs = xf.mode_table(modes, args.grid)
    d = xf.load_set(args.data, modes, args.sci)
    M, gres, ngood = xf.geometry(d, n)
    c, yy, xx, even, amp = d["c"], d["yy"], d["xx"], d["even"], d["amp"]
    ny, nx = d["mref"].shape
    fr = np.hypot(*n.T)
    w = args.radius

    F = d["mref"][np.hypot(xx - c[0], yy - c[1]) < w].sum()
    E = np.full(len(n), np.nan)
    for k in range(len(n)):
        q = M @ n[k]
        if np.hypot(*q) * (1 - args.dl) < 2.5 * w:  # overlaps the core
            continue
        tot = 0.0
        for s in (1, -1):
            qq = c + s * q
            if not (w + 3 < qq[0] < nx - w - 3 and w + 3 < qq[1] < ny - w - 3):
                tot = np.nan
                break
            m = capsule(xx, yy, c, s * q, w, args.dl)
            ann = capsule(xx, yy, c, s * q, w + 5, args.dl + 0.04) & ~capsule(xx, yy, c, s * q, w + 2, args.dl + 0.02)
            e = np.asarray(even[k], dtype=np.float64)
            tot += e[m].sum() - np.median(e[ann]) * m.sum()
        E[k] = tot / F
    G = np.array([solve_G(E[k], amp[k]) for k in range(len(n))])

    print(f"{args.data}: geometry from {ngood} speckles (residual {gres:.2f} px), PSF flux in r < {w:g} px {F:.0f}")
    print(" |n|     N   a [um]   G [rad/um]   g = G/kappa")
    for lo, hi in [(2, 3), (3, 4), (4, 5), (5, 6), (6, 8), (8, 10), (10, 12), (12, 14), (14, 16), (16, 18), (18, 20),
                   (20, 22), (22, 25)]:
        m = (fr >= lo) & (fr < hi) & np.isfinite(G)
        if m.sum() < 3:
            continue
        print(f" {lo:2d}-{hi:<3d} {m.sum():4d}   {np.median(amp[m]):.4f}   {np.median(G[m]):7.2f}      {np.median(G[m]) / args.kappa:.3f}")

    ok = np.isfinite(G)
    f = fr / args.grid
    model = lambda p, f: p[0] * np.exp(-(f / p[1]) ** p[2])  # noqa: E731
    r = least_squares(lambda p: (model(p, f[ok]) - G[ok]) / G[ok], [args.kappa, 0.35, 1.5])
    cov = np.linalg.pinv(r.jac.T @ r.jac) * np.mean(r.fun ** 2)
    se = np.sqrt(np.diag(cov))
    print(f"fit G = kappa0 exp(-(f/f0)^p), f in cycles/actuator: kappa0 {r.x[0]:.2f} +- {se[0]:.2f}, "
          f"f0 {r.x[1]:.4f} +- {se[1]:.4f}, p {r.x[2]:.3f} +- {se[2]:.3f}; rms fractional residual "
          f"{np.sqrt(np.mean(r.fun ** 2)):.3f} (fit over |n| {fr[ok].min():.1f}-{fr[ok].max():.1f})")

    if args.out:
        cols = [fits.Column(name=nm, format="D", array=v) for nm, v in
                [("NX", n[:, 0]), ("NY", n[:, 1]), ("NABS", fr), ("AMP", amp), ("E", E), ("G", G),
                 ("g", G / args.kappa)]]
        hdu = fits.BinTableHDU.from_columns(cols, name="DMRESP")
        h = hdu.header
        h["SRCDATA"] = (args.data.split("/")[-1], "xlinqCal data set")
        h["KAPPA"] = (args.kappa, "4 pi / lambda [rad/um], g = G / KAPPA")
        h["APRAD"] = (w, "aperture radius [px]")
        h["APDL"] = (args.dl, "radial half-elongation of the speckle aperture")
        h["FITK0"], h["FITF0"], h["FITP"] = (r.x[0], "fit kappa0 [rad/um]"), (r.x[1], "fit f0 [cyc/act]"), (r.x[2], "fit p")
        hdu.writeto(args.out, overwrite=True)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
