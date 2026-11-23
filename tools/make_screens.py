#!/usr/bin/env python3
"""Make a random phase-screen poke cube for an xlinqCal cross-term calibration.

Each screen is a random combination of the Fourier modes of the mode cube (1 um rms each):

    screen_s = scale_s * sum_k c_sk m_k,   c_sk ~ N(0, var(|n_k|))   [um, commanded]

so the true phase of every acquisition is known. The variance law var(|n|) is the median per-mode variance
in 1-cycle bins of `--var` (per-mode variances, e.g. the turbulence-residual signal variance measured from
WFS bundles), optionally rescaled to a total rms with `--rms`. Screens are drawn in groups of `--n` per
`--scales` entry, then the first `--repeat` screens of the first group are appended again (noise ceiling
and drift).

Run xlinqCal with `poke.amp=1`. With +/- pokes, (I+ + I-)/2 - ref contains every cross term
sum_kl c_k c_l Re(S_k* S_l), and (I+ - I-)/2 the linear (and third-order) response.

Output: primary = screens (N x 50 x 50, um); extension COEFFS = c (N x Nmode, before scale); extension
SCALE = per-screen scale; extension VARLAW = (|n| bin, var) table; header records the law and the seed.
"""

import argparse
import os
import sys

import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xlq_fourier as xf  # noqa: E402


def var_law(var, fr, width=1.0):
    """Median per-mode variance in |n| bins of `width` cycles, as (bin centers, values), negatives clipped."""
    edges = np.arange(0, fr.max() + width, width)
    cen, val = [], []
    for lo in edges[:-1]:
        m = (fr >= lo) & (fr < lo + width)
        if m.any():
            cen.append(lo + width / 2)
            val.append(max(float(np.median(var[m])), 0.0))
    return np.array(cen), np.array(val)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", required=True, help="the mode cube (1 um rms Fourier modes)")
    ap.add_argument("--var", required=True, help="per-mode variance [um^2] (.npy or FITS), same order as the modes")
    ap.add_argument("--rms", type=float, help="rescale the law to this total rms [um] (default: as measured)")
    ap.add_argument("--n", type=int, default=1200, help="screens per scale")
    ap.add_argument("--scales", default="1,0.5", help="amplitude scales: --n screens at the first, --n-scaled at each other")
    ap.add_argument("--n-scaled", type=int, default=400, help="screens for each scale after the first")
    ap.add_argument("--repeat", type=int, default=100, help="first-group screens appended again at the end")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--grid", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    modes = xf.load_modes(args.modes)
    n, R, th, res, pairs = xf.mode_table(modes, args.grid)
    fr = np.hypot(*n.T)
    var = np.load(args.var) if args.var.endswith(".npy") else fits.getdata(args.var)
    var = np.asarray(var, np.float64).ravel()
    if len(var) != len(modes):
        raise SystemExit(f"--var has {len(var)} values for {len(modes)} modes")

    cen, law = var_law(var, fr)
    v = np.interp(fr, cen, law)
    if args.rms:
        v *= args.rms ** 2 / v.sum()
    print(f"law: total rms {np.sqrt(v.sum()):.4f} um over {len(v)} modes; per-mode rms "
          + " ".join(f"|n|~{c:.0f}: {np.sqrt(np.interp(c, cen, law)) * 1e3:.2f} nm" for c in (2, 5, 10, 15, 20, 25, 30)))

    scales = [float(s) for s in args.scales.split(",")]
    counts = [args.n] + [args.n_scaled] * (len(scales) - 1)
    rng = np.random.default_rng(args.seed)
    C = np.concatenate([rng.standard_normal((k, len(v))) * np.sqrt(v) for k in counts])
    S = np.concatenate([np.full(k, s) for k, s in zip(counts, scales)])
    if args.repeat:
        C = np.concatenate([C, C[:args.repeat]])
        S = np.concatenate([S, S[:args.repeat]])

    flat = modes.reshape(len(modes), -1)
    screens = ((S[:, None] * C) @ flat).reshape(len(C), args.grid, args.grid).astype(np.float32)
    act = np.any(modes != 0, axis=0)
    rms = np.sqrt((screens[:, act] ** 2).mean(1))
    print(f"{len(screens)} screens: rms over active actuators {np.median(rms[S == scales[0]]):.4f} um "
          f"(scale {scales[0]}), max |command| {np.abs(screens).max():.3f} um")

    h = fits.Header()
    h["PRODUCT"] = ("random phase screens of Fourier modes", "run xlinqCal with poke.amp=1")
    h["MODEFILE"] = (os.path.basename(args.modes), "mode cube")
    h["VARFILE"] = (os.path.basename(args.var), "per-mode variance source")
    h["LAWRMS"] = (float(np.sqrt(v.sum())), "total rms of the variance law [um]")
    h["SCALES"] = (args.scales, "amplitude scales of the groups")
    h["NGROUP0"] = (args.n, "screens at the first scale")
    h["NSCALED"] = (args.n_scaled, "screens at each later scale")
    h["NREPEAT"] = (args.repeat, "first screens appended again at the end")
    h["SEED"] = (args.seed, "numpy default_rng seed")
    law_tab = fits.BinTableHDU.from_columns([fits.Column(name="NABS", format="D", array=cen),
                                             fits.Column(name="VAR", format="D", array=law)], name="VARLAW")
    fits.HDUList([fits.PrimaryHDU(screens, header=h), fits.ImageHDU(C.astype(np.float32), name="COEFFS"),
                  fits.ImageHDU(S.astype(np.float32), name="SCALE"), law_tab]).writeto(args.out, overwrite=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
