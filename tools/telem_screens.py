#!/usr/bin/env python3
"""Make an xlinqCal poke cube from on-sky AO telemetry: PCA modes and random screens of the residual.

The input is WFS telemetry recorded at full rate during an observation: the streamWriter `.xrif` archives,
read directly (directories are searched recursively, e.g. `rawimages/aol1_modevalWFS`; `--start`/`--end`
select the observation by atime), or FITS cubes from `xrif2fits -N -C`. Frames are mapped frame by frame to Fourier coefficients c_t
of the unshaped Fourier mode cube (`--modes`, 1 um rms each). `--cal` must be a Fourier-mode xlinqCal run
(the try7 equivalent) taken with the same WFS configuration and reconstructor as the observation.

- `--input modeval`: the loop's modal coefficients `aol1_modevalWFS`, e.g. from a nonlinear (NN)
  reconstructor, through the inverse of the modevalWFS response measured by `--cal`,
  c = R (R^T R + lam)^-1 m, or through a projection of the loop's DM-space modes (`--loop-modes`).
- `--input camwfs`: raw camwfs frames y, calibrated like acquWFS, (y - dark) / sum_wfsmask(y - dark), and
  reconstructed linearly with the imWFS2 response of `--cal`: c = (R R^T + lam)^-1 R y. The reference
  `aol1_wfsref` only shifts the mean, which is removed. Only for a linear WFS (modulated pyramid). On the
  try7 bundles (modulated, linear loop) the modevalWFS route's noise-free DM shapes correlate with this one
  at only 0.94 below 3 cycles/50 act, 0.67 at 6-9 and 0.2-0.4 at 15-24, and this one predicted camsci1.

The 2400 Fourier modes are overcomplete on the ~2040 active actuators, so a DM shape has many Fourier
representations. The modeval routes give the minimum-norm one: the DM shapes and their statistics are
exact (0.99 of the actuator-space variance in a test), but the printed Fourier variances are lower.

WFS noise is white from frame to frame, so the lag-`--lag` covariance (symmetrized) estimates the signal
covariance without the noise. Its eigenvectors are the PCA modes, ordered by signal variance.

Output cube (um of DM surface, run xlinqCal with `poke.amp=1`):

1. `--n-pca` PCA modes, each at `--pca-sigma` times its rms: the +/- pokes give each mode's linear
   response and its own quadratic term.
2. Random screens c ~ N(0, Sigma_signal): `--n` at the first `--scales` entry and `--n-scaled` at each
   other (scales above 1 bracket an underestimated residual, e.g. a linear pyramid's lower optical gain on
   sky). Their even parts sample every cross term in proportion to how
   often the sky produced it.
3. The first `--repeat` screens again (noise ceiling and drift).

`--gain` multiplies every amplitude. The Fourier coefficients of every plane are in extension COEFFS,
so the calibration can be fit in the Fourier basis.
"""

import argparse
import calendar
import glob
import os
import sys
import time

import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xlq_fourier as xf  # noqa: E402
import xrif_read as xr  # noqa: E402


def telem_files(paths, start=None, end=None):
    """xrif (preferred) or FITS files from files and directories (recursive), in file-name time order.

    With `start`/`end` [Unix s], keeps the files that can hold frames in the window (from the last file
    starting before `start`), by the time stamp in the streamWriter file name.
    """
    out = []
    for p in paths:
        if os.path.isdir(p):
            fs = glob.glob(os.path.join(p, "**", "*.xrif"), recursive=True)
            out += fs if fs else glob.glob(os.path.join(p, "**", "*.fits"), recursive=True)
        else:
            out.append(p)
    out = sorted(set(out), key=os.path.basename)
    if start is not None or end is not None:
        t = [xr.file_time(f) for f in out]
        if any(x is None for x in t):
            raise SystemExit("--start/--end need streamWriter file names with time stamps")
        lo = max([i for i, x in enumerate(t) if start is not None and x <= start], default=0)
        out = [f for i, (f, x) in enumerate(zip(out, t)) if i >= lo and (end is None or x <= end)]
    if not out:
        raise SystemExit("no telemetry files found")
    return out


def read_frames(path, start=None, end=None):
    """Frames (T x Npix, float32) and cnt0 (or None for FITS), restricted to atime in [start, end] for xrif."""
    if path.endswith(".xrif"):
        fr, tm = xr.read_xrif(path)
        at = tm[:, 1] + tm[:, 2] * 1e-9
        ok = np.ones(len(at), bool)
        if start is not None:
            ok &= at >= start
        if end is not None:
            ok &= at <= end
        return fr[ok].reshape(ok.sum(), -1).astype(np.float32), tm[ok, 0].astype(np.int64)
    fr = fits.getdata(path)
    return fr.reshape(len(fr), -1).astype(np.float32), None


def accumulate(files, lag, max_frames=0, transform=None, start=None, end=None):
    """Mean, zero-lag covariance and symmetrized lag and 2*lag covariances.

    Lagged pairs are formed within each file, and for xrif only between frames whose cnt0 differ by the lag
    (no pairs across dropped frames). `transform` maps a file's frames (T x Npix, float32) to the
    coefficient vectors (T x K); default identity.
    """
    S = S0 = S1 = S2 = None
    n = n1 = n2 = 0
    for f in files:
        m, cnt = read_frames(f, start, end)
        if max_frames and n + len(m) > max_frames:
            m = m[:max_frames - n]
            cnt = None if cnt is None else cnt[:len(m)]
        if len(m) <= 2 * lag:
            continue
        if transform is not None:
            m = transform(m)
        m = m.astype(np.float64)
        if cnt is None:
            p1 = np.ones(len(m) - lag, bool)
            p2 = np.ones(len(m) - 2 * lag, bool)
        else:
            p1 = cnt[lag:] - cnt[:-lag] == lag
            p2 = cnt[2 * lag:] - cnt[:-2 * lag] == 2 * lag
        if S is None:
            k = m.shape[1]
            S, S0, S1, S2 = np.zeros(k), np.zeros((k, k)), np.zeros((k, k)), np.zeros((k, k))
        elif m.shape[1] != len(S):
            raise SystemExit(f"{f} has {m.shape[1]} modes, earlier files {len(S)}")
        S += m.sum(0)
        S0 += m.T @ m
        S1 += m[:-lag][p1].T @ m[lag:][p1]
        S2 += m[:-2 * lag][p2].T @ m[2 * lag:][p2]
        n, n1, n2 = n + len(m), n1 + p1.sum(), n2 + p2.sum()
        if max_frames and n >= max_frames:
            break
    if S is None:
        raise SystemExit("no usable telemetry frames")
    mu = S / n
    mm = np.outer(mu, mu)
    sym = lambda A: (A + A.T) / 2  # noqa: E731
    return mu, S0 / n - mm, sym(S1 / n1 - mm), sym(S2 / n2 - mm), n


def camwfs_transform(prefix, modes, lam, dark, mask, ref=None):
    """Raw camwfs frames -> Fourier coefficients: acquWFS normalization and the ridge imWFS2 reconstructor."""
    amp = xf.poke_amplitudes(prefix, modes)
    R = fits.getdata(f"{prefix}_aol1_imWFS2pub_resp.fits").reshape(len(amp), -1).astype(np.float64) / amp[:, None]
    G = R @ R.T
    Wp = np.linalg.solve(G + lam * np.trace(G) / len(G) * np.eye(len(G)), R).astype(np.float32)
    d = fits.getdata(dark).astype(np.float32).ravel()
    mk = fits.getdata(mask if mask else f"{prefix}_snap_aol1_wfsmask.fits").ravel() > 0
    r = fits.getdata(ref).astype(np.float32).ravel() if ref else np.zeros(len(d), np.float32)
    if len(d) != R.shape[1] or len(mk) != R.shape[1]:
        raise SystemExit(f"dark ({len(d)}) / mask ({len(mk)}) size differ from the response ({R.shape[1]} px)")

    def f(y):
        if y.shape[1] != len(d):
            raise SystemExit(f"camwfs frames have {y.shape[1]} px, the response {len(d)}")
        y = y - d
        y /= y[:, mk].sum(1, keepdims=True)
        return (y - r) @ Wp.T

    return f, Wp


def map_from_cal(prefix, modes, nmv, lam):
    """W (Nfourier x Nmv): ridge inverse of the measured modevalWFS response per um of each Fourier mode."""
    amp = xf.poke_amplitudes(prefix, modes)
    R = fits.getdata(f"{prefix}_aol1_modevalWFSpub_resp.fits").reshape(len(amp), -1).astype(np.float64) / amp[:, None]
    if R.shape[1] != nmv:
        raise SystemExit(f"--cal response has {R.shape[1]} modevalWFS modes, the telemetry {nmv}: "
                         "the control basis changed, use --loop-modes")
    G = R.T @ R
    return R @ np.linalg.inv(G + lam * np.trace(G) / len(G) * np.eye(len(G)))


def map_from_loop_modes(path, modes, nmv, lam):
    """W (Nfourier x Nmv): ridge least-squares Fourier coefficients of each loop mode's DM shape.

    The 2400 Fourier modes span only ~2011 of the 2039 actuator directions (the rest have singular values ~1e-6),
    so an unregularized fit puts huge, cancelling coefficients on those; the ridge drops them.
    """
    L = fits.getdata(path).astype(np.float64)
    L = L.reshape(len(L), -1)
    if len(L) != nmv:
        raise SystemExit(f"--loop-modes has {len(L)} modes, the telemetry {nmv}")
    F = modes.reshape(len(modes), -1).astype(np.float64)
    act = np.any(F != 0, axis=0)
    Fa, La = F[:, act], L[:, act]
    G = Fa @ Fa.T
    W = np.linalg.solve(G + lam * np.trace(G) / len(G) * np.eye(len(G)), Fa @ La.T)
    res = np.sqrt(((La.T - Fa.T @ W) ** 2).sum() / (La ** 2).sum())
    print(f"loop modes -> Fourier: fractional residual {res:.3f} (loop-mode content the Fourier modes cannot represent)")
    return W


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("telem", nargs="+", help="decoded telemetry FITS cubes, or directories of them")
    ap.add_argument("--input", choices=("modeval", "camwfs"), required=True,
                    help="modeval: loop modal coefficients (NN or linear); camwfs: raw frames, linear WFS only")
    ap.add_argument("--modes", required=True, help="the unshaped Fourier mode cube (1 um rms each)")
    ap.add_argument("--cal", help="xlinqCal prefix of a Fourier-mode run with the same WFS and reconstructor (aol1_modevalWFSpub / aol1_imWFS2pub)")
    ap.add_argument("--loop-modes", help="modeval only: the loop's DM-space modes, Nmode x 50 x 50 [um per unit coefficient]")
    ap.add_argument("--dark", help="camwfs only: the camwfs dark loaded during the observation")
    ap.add_argument("--wfsmask", help="camwfs only: aol1_wfsmask from the observation (default: the --cal snapshot)")
    ap.add_argument("--wfsref", help="camwfs only: aol1_wfsref from the observation (only the static mean MEANC uses it)")
    ap.add_argument("--start", help="first atime to use, UTC ISO 8601 (e.g. 2026-11-25T03:10:00)")
    ap.add_argument("--end", help="last atime to use, UTC ISO 8601")
    ap.add_argument("--every", type=int, default=1, help="use every Nth telemetry file (spread a long observation)")
    ap.add_argument("--lam", type=float, default=1e-3, help="ridge weight of the reconstructor, relative to the mean eigenvalue")
    ap.add_argument("--lag", type=int, default=1, help="lag [frames] of the noise-free covariance")
    ap.add_argument("--max-frames", type=int, default=0, help="use at most this many frames (0: all)")
    ap.add_argument("--n-pca", type=int, default=300, help="PCA modes poked singly")
    ap.add_argument("--pca-sigma", type=float, default=1.0, help="PCA poke amplitude in units of the mode's rms")
    ap.add_argument("--n", type=int, default=1000, help="random screens at the first scale")
    ap.add_argument("--scales", default="1,0.5,1.5", help="amplitude scales: --n screens at the first, --n-scaled at each other")
    ap.add_argument("--n-scaled", type=int, default=250, help="random screens at each scale after the first")
    ap.add_argument("--repeat", type=int, default=100, help="first-group screens appended again at the end")
    ap.add_argument("--gain", type=float, default=1.0, help="multiply every amplitude (e.g. 1/optical gain)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-cmd", type=float, default=1.0, help="refuse to write a cube with a larger |command| [um]")
    ap.add_argument("--exptime", type=float, default=0.036, help="camsci1 exposure [s], for the run-time estimate")
    ap.add_argument("--grid", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    modes = xf.load_modes(args.modes)
    n, *_ = xf.mode_table(modes, args.grid)
    fr = np.hypot(*n.T)
    iso = lambda x: None if x is None else calendar.timegm(time.strptime(x[:19], "%Y-%m-%dT%H:%M:%S"))  # noqa: E731
    t0, t1 = iso(args.start), iso(args.end)
    files = telem_files(args.telem, t0, t1)[::args.every]
    if args.input == "camwfs":
        if not (args.cal and args.dark) or args.loop_modes:
            raise SystemExit("--input camwfs needs --cal and --dark (and not --loop-modes)")
        tf, W = camwfs_transform(args.cal, modes, args.lam, args.dark, args.wfsmask, args.wfsref)
        mu, C0, C1, C2, nfr = accumulate(files, args.lag, args.max_frames, tf, t0, t1)
        print(f"{nfr} camwfs frames from {len(files)} files")
        W = np.eye(len(modes))
    else:
        if bool(args.cal) == bool(args.loop_modes):
            raise SystemExit("--input modeval needs exactly one of --cal and --loop-modes")
        mu, C0, C1, C2, nfr = accumulate(files, args.lag, args.max_frames, None, t0, t1)
        print(f"{nfr} frames of {len(mu)} modes from {len(files)} files")
        W = (map_from_cal(args.cal, modes, len(mu), args.lam) if args.cal
             else map_from_loop_modes(args.loop_modes, modes, len(mu), args.lam))
    Sc0, Sc1, Sc2 = (W @ C @ W.T for C in (C0, C1, C2))
    muc = W @ mu
    v0, v1, v2 = np.diag(Sc0), np.diag(Sc1), np.diag(Sc2)
    print(f"Fourier rms: total {np.sqrt(v0.sum()):.4f} um, signal (lag {args.lag}) {np.sqrt(max(v1.sum(), 0)):.4f} um, "
          f"noise {np.sqrt(max(v0.sum() - v1.sum(), 0)):.4f} um, static mean {np.sqrt((muc ** 2).sum()):.4f} um")
    print(" |n|   signal rms [nm]  noise rms [nm]  lag2/lag1")
    for lo in range(0, int(fr.max()) + 1, 4):
        b = (fr >= lo) & (fr < lo + 4)
        if b.any():
            s1, s0, s2 = v1[b].sum(), v0[b].sum(), v2[b].sum()
            print(f" {lo:2d}-{lo + 4:<2d}  {np.sqrt(max(s1, 0) / b.sum()) * 1e3:13.2f}  "
                  f"{np.sqrt(max(s0 - s1, 0) / b.sum()) * 1e3:14.2f}  {s2 / s1 if s1 > 0 else np.nan:9.3f}")

    lam_e, V = np.linalg.eigh(Sc1)
    o = np.argsort(lam_e)[::-1]
    lam_e, V = lam_e[o], V[:, o]
    pos = lam_e > 0
    lam_p = np.where(pos, lam_e, 0.0)
    npca = min(args.n_pca, int(pos.sum()))
    frac = np.cumsum(lam_p) / lam_p.sum()
    print(f"PCA: {pos.sum()} positive eigenvalues; the first {npca} hold {frac[npca - 1] * 100:.1f}% of the signal variance "
          f"(100: {frac[min(99, len(frac) - 1)] * 100:.1f}%, 500: {frac[min(499, len(frac) - 1)] * 100:.1f}%)")

    rng = np.random.default_rng(args.seed)
    Cp = (V[:, :npca] * (args.pca_sigma * np.sqrt(lam_p[:npca]))).T          # PCA pokes (npca x Nf)
    scales = [float(s) for s in args.scales.split(",")]
    counts = [args.n] + [args.n_scaled] * (len(scales) - 1)
    Z = rng.standard_normal((sum(counts), int(pos.sum())))
    Cs = (Z * np.sqrt(lam_p[pos])) @ V[:, pos].T                                 # unit-scale screens
    Ss = np.concatenate([np.full(k, s) for k, s in zip(counts, scales)])
    Cs *= Ss[:, None]
    if args.repeat:
        Cs = np.concatenate([Cs, Cs[:args.repeat]])
        Ss = np.concatenate([Ss, Ss[:args.repeat]])
    C = args.gain * np.concatenate([Cp, Cs])
    kind = np.concatenate([np.zeros(npca), np.ones(sum(counts)), np.full(args.repeat, 2)]).astype(np.int16)
    scale = np.concatenate([np.full(npca, args.pca_sigma), Ss]) * args.gain

    screens = (C @ modes.reshape(len(modes), -1)).reshape(len(C), args.grid, args.grid).astype(np.float32)
    act = np.any(modes != 0, axis=0)
    rms = np.sqrt((screens[:, act] ** 2).mean(1))
    cmax = float(np.abs(screens).max())
    print(f"{len(screens)} planes: {npca} PCA modes, {sum(counts)} screens (median rms at scale {scales[0]}: "
          f"{np.median(rms[npca:npca + args.n]):.4f} um), "
          f"{args.repeat} repeats; max |command| {cmax:.3f} um")
    print(f"xlinqCal run time ~{len(screens) * 6 * (args.exptime + 0.017) / 60:.0f} min at {args.exptime * 1e3:.0f} ms "
          "(+, -, ref per plane, one settle frame each)")
    if cmax > args.max_cmd:
        raise SystemExit(f"max |command| {cmax:.3f} um is above --max-cmd {args.max_cmd}: not written")

    h = fits.Header()
    h["PRODUCT"] = ("AO residual PCA modes and screens", "use poke.amp=1")
    h["MODEFILE"] = (os.path.basename(args.modes), "Fourier mode cube of COEFFS")
    h["INPUT"] = (args.input, "telemetry stream type")
    h["MAPSRC"] = (os.path.basename(args.cal or args.loop_modes), "telemetry -> Fourier map source")
    h["MAPLAM"] = (args.lam, "ridge weight of the reconstructor")
    h["DARK"] = (os.path.basename(args.dark or ""), "camwfs dark")
    h["WFSMASK"] = (os.path.basename(args.wfsmask or ""), "camwfs normalization mask (empty: cal snapshot)")
    h["WFSREF"] = (os.path.basename(args.wfsref or ""), "camwfs reference (empty: none, MEANC not meaningful)")
    h["TSTART"] = (args.start or "", "first atime used (UTC)")
    h["TEND"] = (args.end or "", "last atime used (UTC)")
    h["EVERY"] = (args.every, "telemetry file stride")
    h["TELNFR"] = (nfr, "telemetry frames used")
    h["TELNFIL"] = (len(files), "telemetry files")
    h["TELFIRST"] = (os.path.basename(files[0]), "first file")
    h["TELLAST"] = (os.path.basename(files[-1]), "last file")
    h["LAG"] = (args.lag, "lag of the signal covariance [frames]")
    h["SIGRMS"] = (float(np.sqrt(max(v1.sum(), 0))), "signal rms of the residual [um]")
    h["NOISRMS"] = (float(np.sqrt(max(v0.sum() - v1.sum(), 0))), "WFS noise rms per frame [um]")
    h["NPCA"] = (npca, "PCA mode planes first")
    h["PCASIG"] = (args.pca_sigma, "PCA poke amplitude [mode rms]")
    h["PCAFRAC"] = (float(frac[npca - 1]), "signal variance fraction of the PCA modes")
    h["SCALES"] = (args.scales, "screen amplitude scales")
    h["NGROUP0"] = (args.n, "screens at the first scale")
    h["NSCALED"] = (args.n_scaled, "screens at each later scale")
    h["NREPEAT"] = (args.repeat, "first screens appended again at the end")
    h["GAIN"] = (args.gain, "multiplier on every amplitude")
    h["SEED"] = (args.seed, "numpy default_rng seed")
    hd = [fits.PrimaryHDU(screens, header=h),
          fits.ImageHDU(C.astype(np.float32), name="COEFFS"),
          fits.ImageHDU(kind, name="KIND"),
          fits.ImageHDU(scale.astype(np.float32), name="SCALE"),
          fits.ImageHDU(lam_e.astype(np.float64), name="EIGVAL"),
          fits.ImageHDU(V[:, :npca].T.astype(np.float32), name="EIGVEC"),
          fits.ImageHDU(np.c_[v0, v1, v2].T.astype(np.float64), name="VARC"),
          fits.ImageHDU(muc.astype(np.float64), name="MEANC")]
    if args.input == "modeval":
        hd.append(fits.ImageHDU(W.astype(np.float32), name="WMAP"))
    fits.HDUList(hd).writeto(args.out, overwrite=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
