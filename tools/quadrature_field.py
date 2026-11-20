#!/usr/bin/env python3
"""Estimate the coherent focal-plane field C from cos/sin pairs, and reconstruct the pupil field.

For each Fourier frequency n, the cos and sin modes put fields in quadrature at each speckle, so their
odd parts give both quadratures of C there (pairwise probing). At the speckle peak pixel:

    C = conj(Z) / (2 sqrt(e)),   e = mean(even_k / a_k^2) of the pair,

which needs no kappa (the DM-to-phase gain). Outputs:

- the coherent fraction R = |C|^2 / ref by radius: the part of the reference light that interferes with
  the probes. |Z|^2 = o_cos^2 + o_sin^2, so R doesn't depend on any phase convention;
- the pupil-plane field: C(n) on the frequency lattice, inverse FFT on the actuator grid, with the
  missing n = 0 (core) term fitted so the field is zero outside the pupil and in the obscuration.
  Amplitude shows the pupil registered on the DM; phase is the quasi-static aberration seen by camsci1.

The phase orientation (which speckle of a pair is e^{+i psi}) is set by --side-sign; the 2f test
(tools/test_2f.py) measures it. With the wrong sign the pupil field is conj(E(-x)) (the twin image).

Output FITS: primary = [Re E, Im E, |E|, arg E] (4 x grid x grid), extension POINTS = one row per
speckle: n_x, n_y, x, y, r [px], Re C, Im C, e, Z noise bias, ref, R.
"""

import argparse

import numpy as np
from astropy.io import fits

import xlq_fourier as xf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", required=True, help="the mode cube (1 um rms modes)")
    ap.add_argument("--data", required=True, help="data set prefix, e.g. try2/xlinqCal_20261119_002")
    ap.add_argument("--sci", default="camsci1", help="science stream name in the output files")
    ap.add_argument("--grid", type=int, default=50, help="actuators across the mode grid")
    ap.add_argument("--side-sign", type=int, default=-1, choices=[-1, 1],
                    help="e^{+i psi} speckle at c + side_sign*M@n; -1 was measured on try1/try2")
    ap.add_argument("--out", required=True, help="output FITS file")
    ap.add_argument("--png", help="optional PNG of the pupil amplitude and phase")
    args = ap.parse_args()

    modes = xf.load_modes(args.modes)
    n, R, th, res, pairs = xf.mode_table(modes, args.grid)
    if res.max() > 1e-3 or any(len(v) != 2 for v in pairs.values()):
        raise ValueError("modes are not cos/sin pairs of pure sinusoids")

    d = xf.load_set(args.data, modes, args.sci)
    M, gres, ngood = xf.geometry(d, n)
    resp, even, mref, amp, c, sig = d["resp"], d["even"], d["mref"], d["amp"], d["c"], d["sig"]
    ny, nx = mref.shape

    pts = []
    for key, (k1, k2) in pairs.items():
        nn = np.array(key, float)
        for s in (1, -1):
            p = c + s * args.side_sign * (M @ nn)
            x0, y0 = int(round(p[0])), int(round(p[1]))
            if not (3 <= x0 < nx - 3 and 3 <= y0 < ny - 3):
                continue
            sub = (even[k1, y0 - 2:y0 + 3, x0 - 2:x0 + 3] / amp[k1] ** 2
                   + even[k2, y0 - 2:y0 + 3, x0 - 2:x0 + 3] / amp[k2] ** 2)
            dy, dx = np.unravel_index(np.argmax(sub), sub.shape)
            y, x = y0 - 2 + dy, x0 - 2 + dx
            Z = xf.solve_Z(resp[k1, y, x] / amp[k1], resp[k2, y, x] / amp[k2], th[k1], th[k2], s)
            e = 0.5 * (even[k1, y, x] / amp[k1] ** 2 + even[k2, y, x] / amp[k2] ** 2)
            bias = (sig[k1] / amp[k1]) ** 2 + (sig[k2] / amp[k2]) ** 2
            C = np.conj(Z) / (2 * np.sqrt(max(e, 1e-12)))
            Rc = (abs(Z) ** 2 - bias) / (4 * e * mref[y, x]) if mref[y, x] > 0 and e > 0 else np.nan
            pts.append([s * nn[0], s * nn[1], x, y, np.hypot(x - c[0], y - c[1]), C.real, C.imag, e, bias,
                        mref[y, x], Rc])
    pts = np.array(pts)

    r, Rc = pts[:, 4], pts[:, 10]
    snr = np.sqrt((pts[:, 5] ** 2 + pts[:, 6] ** 2) * 4 * pts[:, 7] / pts[:, 8])
    print(f"PSF center ({c[0]:.2f}, {c[1]:.2f}); geometry from {ngood} modes, residual {gres:.2f} px; "
          f"{len(pts)} speckles")
    print(" radius [px] |    N | ref [counts] | |Z| SNR | coherent fraction R: median [IQR]")
    for lo, hi in [(0, 12), (12, 20), (20, 30), (30, 45), (45, 65), (65, 90), (90, 130), (130, 170)]:
        m = (r >= lo) & (r < hi)
        if np.isfinite(Rc[m]).any():
            q = np.nanpercentile(Rc[m], [25, 50, 75])
            print(f"   {lo:3d}-{hi:<4d} | {m.sum():4d} | {np.median(pts[m, 9]):12.1f} | {np.median(snr[m]):7.1f} | "
                  f"{q[1]:.3f} [{q[0]:.3f}, {q[2]:.3f}]")

    # Pupil field: lattice -> inverse FFT, then fit the n = 0 term
    g = args.grid
    Cg = np.zeros((g, g), complex)
    for (mx, my), cr, ci in zip(pts[:, :2].astype(int), pts[:, 5], pts[:, 6]):
        Cg[my % g, mx % g] = cr + 1j * ci
    E = np.fft.ifft2(Cg) * g * g

    rr, cc = np.mgrid[0:g, 0:g]
    ann = np.abs(np.angle(E)) > np.pi / 2  # without the core term the illuminated annulus is near phase pi
    y0, x0 = rr[ann].mean(), cc[ann].mean()
    rad = np.hypot(rr - y0, cc - x0)
    zero = (rad > 0.51 * g) | (rad < 4.5)
    E = E - E[zero].mean()
    a = np.abs(E)
    top = np.median(a[(rad > 0.2 * g) & (rad < 0.4 * g)])
    prof = np.array([np.median(a[(rad >= b) & (rad < b + 1)]) for b in np.arange(0, g // 2 + 2, 0.5)])
    bins = np.arange(0, g // 2 + 2, 0.5)
    outer = bins[np.where((prof < top / 2) & (bins > 0.3 * g))[0][0]] + 0.5
    inner = bins[np.where(prof > top / 2)[0][0]] + 0.5
    inside = ~zero & (a > 0.5 * top) & (rad < outer)
    ph = np.angle(E * np.exp(-1j * np.angle(E[inside].mean())))
    print(f"pupil center (col, row) = ({x0:.2f}, {y0:.2f}) act; half-max diameter ~{2 * outer:.0f} act, "
          f"obscuration ~{2 * inner:.0f} act; phase rms inside {ph[inside].std():.3f} rad; "
          f"outside/inside amplitude {a[zero].mean() / a[inside].mean():.2f}")

    hdr = fits.Header()
    hdr["SRCDATA"] = (args.data.split("/")[-1], "xlinqCal data set")
    hdr["SIDESGN"] = (args.side_sign, "e^{+i psi} speckle at c + SIDESGN*M@n")
    hdr["PUPX0"] = (x0, "pupil center column [act]")
    hdr["PUPY0"] = (y0, "pupil center row [act]")
    for (i, j), v in np.ndenumerate(M):
        hdr[f"GEOM{i}{j}"] = (v, "camsci1 px = GEOM @ (n_x, n_y)")
    hdr["COMMENT"] = "Planes: Re E, Im E, |E|, arg E (arg relative to the pupil mean) on the actuator grid."
    cube = np.stack([E.real, E.imag, a, np.where(inside, ph, np.nan)]).astype(np.float32)
    fits.HDUList([fits.PrimaryHDU(cube, header=hdr), fits.ImageHDU(pts, name="POINTS")]).writeto(args.out,
                                                                                               overwrite=True)
    print(f"wrote {args.out}")

    if args.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
        im = ax[0].imshow(a, origin="lower")
        ax[0].set_title("|E| (pupil on DM)")
        plt.colorbar(im, ax=ax[0])
        im = ax[1].imshow(np.where(inside, ph, np.nan), origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
        ax[1].set_title("phase [rad]")
        plt.colorbar(im, ax=ax[1])
        plt.tight_layout()
        plt.savefig(args.png, dpi=80)


if __name__ == "__main__":
    main()
