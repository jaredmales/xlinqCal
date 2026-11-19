#!/usr/bin/env python3
"""Make an amplitude-shaped Fourier poke cube from a previous xlinqCal data set.

Each mode's amplitude is chosen so that its camsci1 speckle pair (the even part, |S|^2) carries
about `--target` of the PSF flux, capped at `--cap` microns rms:

    amp_k = amp0 * sqrt(target / E(|n_k|)),   amp_k <= cap

where amp0 is the data set's POKEAMP and E(|n|) is a smooth fit (cubic in |n|, in log) to the
measured speckle energy per mode. The output cube is `amp_k * mode_k` (microns), so run xlinqCal
with `poke.amp=1`. Extensions AMP (per-mode amplitude) and FREQ (n_x, n_y in cycles per grid) are
appended, and the header records the law and the fitted camsci1 geometry.

Steps: (1) mode frequencies from the FFT of each mode, (2) speckle peak per mode in the even cube,
(3) linear map from frequency to camsci1 pixels (up to sign), (4) speckle energy per mode,
(5) amplitude law.
"""

import argparse

import numpy as np
from astropy.io import fits


def mode_frequencies(modes, grid):
    """Return integer (n_x, n_y) cycles per `grid` actuators for each mode, from the FFT peak."""
    npad = grid * 96
    freqs = np.zeros((len(modes), 2))

    for k, m in enumerate(modes):
        F = np.abs(np.fft.rfft2(m, (npad, npad)))
        F[0, 0] = 0
        fy, fx = divmod(np.argmax(F), F.shape[1])
        fy = fy if fy < npad / 2 else fy - npad
        freqs[k] = [fx * grid / npad, fy * grid / npad]

    n = np.round(freqs)
    if np.abs(freqs - n).max() > 0.05:
        raise ValueError("mode frequencies are not integer cycles per grid; check --grid")

    return n


def even_peaks(even, cx, cy, core):
    """Return (dx, dy, peak, robust noise) of the brightest off-core pixel in each even image."""
    yy, xx = np.mgrid[0:even.shape[1], 0:even.shape[2]]
    incore = np.hypot(xx - cx, yy - cy) < core
    out = np.zeros((even.shape[0], 4))

    for k in range(even.shape[0]):
        e = np.asarray(even[k], dtype=np.float64)
        y, x = divmod(np.argmax(np.where(incore, -np.inf, e)), e.shape[1])
        bg = e[~incore]
        out[k] = [x - cx, y - cy, e[y, x], 1.4826 * np.median(np.abs(bg - np.median(bg)))]

    return out


def fit_geometry(n, peaks, snr_min=20, nmin=3, nmax=22):
    """Fit pix = M @ n to the speckle peaks, allowing either speckle of the pair (sign)."""
    snr = peaks[:, 2] / peaks[:, 3]
    fr = np.hypot(*n.T)
    good = (snr > snr_min) & (fr > nmin) & (fr < nmax)
    X, Y = n[good], peaks[good, :2]
    M = np.eye(2) * 4.0

    for _ in range(30):
        sgn = np.sign(np.sum((X @ M.T) * Y, 1))
        sgn[sgn == 0] = 1
        Ys = Y * sgn[:, None]
        M = np.linalg.lstsq(X, Ys, rcond=None)[0].T

    resid = np.sqrt(np.mean(np.sum((Ys - X @ M.T) ** 2, 1)))

    return M, resid, int(good.sum())


def speckle_energy(even, n, M, cx, cy, psf_total, aperture=6.0):
    """Return the even-part flux in both speckles of each mode, as a fraction of the PSF flux."""
    yy, xx = np.mgrid[0:even.shape[1], 0:even.shape[2]]
    E = np.zeros(len(n))

    for k in range(len(n)):
        px, py = n[k] @ M.T
        rad = min(aperture, 0.45 * np.hypot(px, py))
        mask = np.zeros(xx.shape, bool)

        for s in (1, -1):
            mask |= np.hypot(xx - (cx + s * px), yy - (cy + s * py)) < rad

        E[k] = np.asarray(even[k], dtype=np.float64)[mask].sum() / psf_total

    return E


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", required=True, help="the mode cube used for the data set (1 um rms modes)")
    ap.add_argument("--data", required=True, help="data set prefix, e.g. try1/xlinqCal_20261119_001")
    ap.add_argument("--sci", default="camsci1", help="science stream name in the output files")
    ap.add_argument("--grid", type=int, default=50, help="actuators across the grid the modes are defined on")
    ap.add_argument("--target", type=float, default=0.01, help="target speckle energy fraction per mode")
    ap.add_argument("--cap", type=float, default=0.05, help="maximum amplitude [um rms]")
    ap.add_argument("--out", required=True, help="output FITS file")
    args = ap.parse_args()

    modes = fits.getdata(args.modes).astype(np.float64)
    even = fits.getdata(f"{args.data}_{args.sci}_even.fits", memmap=True)
    ehead = fits.getheader(f"{args.data}_{args.sci}_even.fits")
    amp0, poke0 = ehead["POKEAMP"], ehead["POKE0"]

    if poke0 != 0 or even.shape[0] != len(modes):
        raise ValueError("the data set must cover all modes of the mode cube, starting at 0")

    ref = fits.getdata(f"{args.data}_{args.sci}_ref.fits", memmap=True)
    mref = np.mean(np.asarray(ref[: min(len(ref), 200)], dtype=np.float64), 0)
    border = np.concatenate([mref[:30].ravel(), mref[-30:].ravel()])
    bias = np.median(border)
    py, px = np.unravel_index(np.argmax(mref), mref.shape)
    win = np.clip(mref[py - 3:py + 4, px - 3:px + 4] - bias, 0, None)
    wy, wx = np.mgrid[-3:4, -3:4]
    cx, cy = px + (win * wx).sum() / win.sum(), py + (win * wy).sum() / win.sum()
    psf_total = (mref - bias).sum()

    n = mode_frequencies(modes, args.grid)
    fr = np.hypot(*n.T)
    peaks = even_peaks(even, cx, cy, core=8)
    M, resid, ngood = fit_geometry(n, peaks)
    E = speckle_energy(even, n, M, cx, cy, psf_total)

    ok = (E > 0) & (fr >= 2)
    coef = np.polyfit(fr[ok], np.log(E[ok]), 3)
    Es = np.exp(np.polyval(coef, fr))
    amp = np.minimum(amp0 * np.sqrt(args.target / Es), args.cap)

    sv = np.linalg.svd(M, compute_uv=False)
    rot = np.rad2deg(np.arctan2(M[1, 0], M[0, 0]))
    print(f"PSF center ({cx:.2f}, {cy:.2f}), bias {bias:.1f}; geometry from {ngood} modes: "
          f"{sv[0]:.3f}/{sv[1]:.3f} px per cycle/{args.grid} act, rotation {rot:.2f} deg, residual {resid:.2f} px")
    print("|n| bin : measured E at amp0, new amp [um]")
    for lo, hi in [(1, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 40)]:
        m = (fr >= lo) & (fr < hi)
        if m.any():
            print(f"  {lo:2d}-{hi:2d}: E {np.median(E[m]):.4f}  amp {np.median(amp[m]):.4f}")

    hdr = fits.Header()
    hdr["SHAPESRC"] = (args.data.split("/")[-1], "xlinqCal data set used for the law")
    hdr["SHAPEMOD"] = (args.modes.split("/")[-1], "mode cube (1 um rms modes)")
    hdr["SHAPEA0"] = (amp0, "POKEAMP of the source data set [um]")
    hdr["SHAPETGT"] = (args.target, "target speckle energy fraction per mode")
    hdr["SHAPECAP"] = (args.cap, "amplitude cap [um rms]")
    hdr["SHAPEGRD"] = (args.grid, "freqs in cycles per this many actuators")
    for i, c in enumerate(coef):
        hdr[f"SHAPEC{i}"] = (c, "log E(|n|) cubic coef, highest first")
    for (i, j), v in np.ndenumerate(M):
        hdr[f"GEOM{i}{j}"] = (v, "camsci1 px = GEOM @ (n_x, n_y)")
    hdr["GEOMRES"] = (resid, "geometry fit residual [px]")
    hdr["COMMENT"] = "Planes are amp_k * mode_k in um; run xlinqCal with poke.amp=1."

    shaped = (modes * amp[:, None, None]).astype(np.float32)
    hdus = [fits.PrimaryHDU(shaped, header=hdr),
            fits.ImageHDU(amp, name="AMP"),
            fits.ImageHDU(n, name="FREQ")]
    fits.HDUList(hdus).writeto(args.out, overwrite=True)
    print(f"wrote {args.out}: {len(modes)} modes, amp {amp.min():.4f}-{amp.max():.4f} um")


if __name__ == "__main__":
    main()
