#!/usr/bin/env python3
"""Flux-calibrate an xlinqCal PSF block with its calibration speckles.

A PSF block taken with the wheel moved and psf.speckAmp > 0 has 6 planes: at the PSF filter the DM
reference, +speckles, -speckles, then the same at the calibration filter (ORIGFILT). At each filter
(+ + -)/2 - reference is the speckle intensity without interference with the static field (and without
the dark), so the ratio of the speckle fluxes is the transmission T of the PSF filter relative to the
calibration filter. The PSF in calibration-filter counts is then (reference - dark) / T. T also joins
the two segments of a run with psf.unsatLast (xlq_fourier.load_set uses it).

The speckles are the brightest local maxima of the PSF-filter speckle image outside --rmin of the core,
measured in round apertures with an annulus background. A speckle is dropped if any of its pixels is
above --linear-max in a calibration-filter plane. With an in-place block (3 planes, no wheel move) the
speckle fluxes are printed, for comparison with a block from another run.
"""

import argparse
import os

import numpy as np
from astropy.io import fits
from scipy.ndimage import maximum_filter, uniform_filter


def phot(img, x, y, r):
    """Aperture sum within r of (x, y), less the median of the annulus r + 2 to r + 6, and the pixel mask."""
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    rr = np.hypot(xx - x, yy - y)
    ap = rr < r
    ann = (rr > r + 2) & (rr < r + 6)
    return img[ap].sum() - np.median(img[ann]) * ap.sum(), ap


def measure(block, nspeck=None, radius=5.0, rmin=15.0, linear_max=37000):
    """Measure the calibration speckles of a PSF block.

    Returns a dict: cube, header, core (x, y), filt, orig, moved, rows [(x, y, flux_psf, flux_cal, linear)],
    and for a moved block T (flux ratio over the linear speckles), Tse (standard error of the per-speckle
    ratios) and n (speckles used); T is NaN if none is linear.
    """
    cube = fits.getdata(block).astype(np.float64)
    h = fits.getheader(block)
    if cube.ndim == 2 or cube.shape[0] not in (3, 6):
        raise ValueError(f"{block}: needs a block with calibration speckles (3 or 6 planes)")
    nsp = nspeck or (4 if h.get("SPKCROSS", 1) else 2)

    spk = 0.5 * (cube[1] + cube[2]) - cube[0]
    ref = cube[0]
    py, px = np.unravel_index(np.argmax(uniform_filter(ref, 3)), ref.shape)
    if "SATCX" in h:  # the core mask center (first, second image axis) is the better center when saturated
        px, py = h["SATCX"], h["SATCY"]

    yy, xx = np.mgrid[0:ref.shape[0], 0:ref.shape[1]]
    sm = uniform_filter(spk, 3)
    edge = radius + 7
    cand = (sm == maximum_filter(sm, size=int(2 * radius + 1))) & (np.hypot(xx - px, yy - py) > rmin)
    cand &= (xx > edge) & (xx < ref.shape[1] - edge) & (yy > edge) & (yy < ref.shape[0] - edge)
    cy, cx = np.nonzero(cand)
    order = np.argsort(sm[cy, cx])[::-1][:nsp]
    if len(order) < nsp:
        raise ValueError(f"{block}: found only {len(order)} speckle candidates")

    moved = cube.shape[0] == 6
    spc = 0.5 * (cube[4] + cube[5]) - cube[3] if moved else None
    rows = []
    for i in order:
        x, y = cx[i], cy[i]
        f1, apm = phot(spk, x, y, radius)
        f2, lin = np.nan, True
        if moved:
            f2, _ = phot(spc, x, y, radius)
            lin = max(cube[p][apm].max() for p in (3, 4, 5)) <= linear_max
        rows.append((x, y, f1, f2, lin))

    out = dict(cube=cube, header=h, core=(px, py), filt=str(h.get("PSFFILT", "?")).strip(),
               orig=str(h.get("ORIGFILT", "?")).strip(), moved=moved, rows=rows,
               T=np.nan, Tse=np.nan, n=0)
    if moved:
        fp = np.array([r[2] for r in rows if r[4]])
        fc = np.array([r[3] for r in rows if r[4]])
        if len(fp):
            r = fp / fc
            out.update(T=fp.sum() / fc.sum(), Tse=np.std(r, ddof=1) / np.sqrt(len(r)) if len(r) > 1 else np.nan,
                       n=len(r))
    return out


def run_transmission(prefix, sci="camsci1", **kw):
    """T from `<prefix>_<sci>_psf.fits` and `_psf_end.fits` (the mean of those present and measured).

    Returns (T, list of (file, T, Tse, n)). Raises if no block gives T.
    """
    res = []
    for suf in ("_psf.fits", "_psf_end.fits"):
        f = f"{prefix}_{sci}{suf}"
        if os.path.exists(f):
            m = measure(f, **kw)
            if m["moved"] and np.isfinite(m["T"]):
                res.append((f, m["T"], m["Tse"], m["n"]))
    if not res:
        raise ValueError(f"no PSF block with calibration speckles at both filters for {prefix}")
    return float(np.mean([r[1] for r in res])), res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("block", help="the PSF block, e.g. <prefix>_camsci1_psf.fits")
    ap.add_argument("--nspeck", type=int, help="speckles to measure (default 4 if SPKCROSS, else 2)")
    ap.add_argument("--radius", type=float, default=5.0, help="aperture radius [px]")
    ap.add_argument("--rmin", type=float, default=15.0, help="ignore maxima within this of the core [px]")
    ap.add_argument("--linear-max", type=float, default=37000, help="camsci1 linearity limit [counts]")
    ap.add_argument("--dark", help="dark frame (default <prefix>_<sci>_dark.fits next to the block, if present)")
    ap.add_argument("--out", help="write the PSF in calibration-filter counts here")
    args = ap.parse_args()

    try:
        m = measure(args.block, args.nspeck, args.radius, args.rmin, args.linear_max)
    except ValueError as e:
        raise SystemExit(str(e))
    filt, orig, (px, py) = m["filt"], m["orig"], m["core"]
    print(f"{os.path.basename(args.block)}: {m['cube'].shape[0]} planes, core at ({px:.1f}, {py:.1f}), "
          f"speckles at {filt}" + (f" and {orig}" if m["moved"] else " (in place)"))
    print("    x       y      r    flux@" + filt + ("   flux@" + orig + "   ratio" if m["moved"] else ""))
    for x, y, f1, f2, lin in m["rows"]:
        line = f" {x:6.1f}  {y:6.1f}  {np.hypot(x - px, y - py):5.1f}  {f1:10.1f}"
        if m["moved"]:
            line += f"  {f2:10.1f}  {f1 / f2:.5f}" + ("" if lin else "  nonlinear: dropped")
        print(line)

    if not m["moved"]:
        return
    if m["n"] == 0:
        raise SystemExit("no speckle is linear at the calibration filter: lower psf.speckAmp")

    T, se = m["T"], m["Tse"]
    print(f"transmission {filt}/{orig} T = {T:.5f} (+- {se:.5f} from the per-speckle scatter, {m['n']} speckles), "
          f"{-2.5 * np.log10(T):.3f} mag")

    if args.out:
        ref = m["cube"][0]
        dark = args.dark
        if dark is None:
            pre = args.block.rsplit("_psf", 1)[0]
            dark = pre + "_dark.fits" if os.path.exists(pre + "_dark.fits") else None
        if dark:
            ref = ref - fits.getdata(dark).astype(np.float64)
        else:
            ref = ref - np.median(np.concatenate([ref[:30].ravel(), ref[-30:].ravel()]))
        out = fits.PrimaryHDU((ref / T).astype(np.float32))
        o = out.header
        o["PRODUCT"] = "PSF at PSFFILT in ORIGFILT counts: (ref - dark) / TRANS"
        o["SRCBLOCK"] = (os.path.basename(args.block), "PSF block")
        o["PSFFILT"], o["ORIGFILT"] = filt, orig
        o["TRANS"] = (T, "speckle flux ratio PSFFILT / ORIGFILT")
        o["TRANSERR"] = (se, "standard error of the per-speckle ratios")
        o["NSPECK"] = (m["n"], "speckles used")
        o["DARK"] = (os.path.basename(dark) if dark else "edge median", "dark subtracted")
        out.writeto(args.out, overwrite=True)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
