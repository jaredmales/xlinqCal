"""Shared helpers for analyzing xlinqCal Fourier-mode data sets.

Conventions (see planning/linq_analysis.md sections 9.11-9.12):

- Modes are pure sinusoids on the 50-actuator grid, masked to the active actuators:
  m_k = R cos(psi + theta_k), psi = 2 pi (n_x col + n_y row) / grid, R = sqrt(2).
  n is put in canonical form (n_x > 0, or n_x == 0 and n_y > 0), so the two modes at each n are a
  cos/sin pair in exact quadrature.
- camsci1 geometry: speckle offset (px) = +-M @ n, with M normalized to a positive trace. The speckle
  carrying the e^{+i psi} component is at c + side_sign * M @ n. The 2f test measured side_sign = -1
  on try1 and try2.
- odd part o_k = resp_k / a_k (per um). For a cos/sin pair, Z = sqrt(2) kappa C* A at each pixel near
  the speckle, from  o/a = -s sin(theta) Re Z - cos(theta) Im Z  (s = +1 at the e^{+i psi} speckle,
  -1 at the other).
"""

import os
import sys

import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shape_pokes as sp  # noqa: E402


def mode_table(modes, grid=50):
    """Return canonical n (N x 2), amplitude R, phase theta, fit residual rms, and {tuple(n): [k1, k2]}."""
    F = np.abs(np.fft.fft2(modes))
    F[:, 0, 0] = 0
    fy, fx = np.divmod(F.reshape(len(modes), -1).argmax(1), grid)
    fy = np.where(fy > grid // 2, fy - grid, fy)
    fx = np.where(fx > grid // 2, fx - grid, fx)
    n = np.c_[fx, fy].astype(float)
    flip = (n[:, 0] < 0) | ((n[:, 0] == 0) & (n[:, 1] < 0))
    n[flip] *= -1

    row, col = np.mgrid[0:grid, 0:grid]
    R = np.zeros(len(n))
    th = np.zeros(len(n))
    res = np.zeros(len(n))

    for k in range(len(n)):
        sup = modes[k] != 0
        psi = 2 * np.pi * (n[k, 0] * col[sup] + n[k, 1] * row[sup]) / grid
        A = np.c_[np.cos(psi), np.sin(psi)]
        (a, b), *_ = np.linalg.lstsq(A, modes[k][sup], rcond=None)
        R[k], th[k] = np.hypot(a, b), np.arctan2(-b, a)
        res[k] = np.std(modes[k][sup] - A @ [a, b])

    pairs = {}
    for k in range(len(n)):
        pairs.setdefault(tuple(n[k].astype(int)), []).append(k)

    return n, R, th, res, pairs


def poke_amplitudes(prefix, modes, sci="camsci1"):
    """Per-mode amplitude [um]: projection of `<prefix>_pokes.fits` onto the modes, else POKEAMP."""
    pf = f"{prefix}_pokes.fits"
    if os.path.exists(pf):
        p = fits.getdata(pf).astype(np.float64)
        return np.einsum("kij,kij->k", p, modes) / np.einsum("kij,kij->k", modes, modes)

    h = fits.getheader(f"{prefix}_{sci}_even.fits")
    return np.full(len(modes), float(h["POKEAMP"]))


def load_set(prefix, modes, sci="camsci1"):
    """Load resp, even, mean bias-subtracted ref, PSF center, amplitudes, and per-mode resp noise.

    The ref is dark-subtracted with `<prefix>_<sci>_dark.fits` when it exists, otherwise with the median
    of the top and bottom 30 rows. The center is the centroid around the ref maximum; with a focal-plane
    mask, refit it with geometry(..., fit_center=True).
    """
    h = fits.getheader(f"{prefix}_{sci}_even.fits")
    if int(h["POKE0"]) != 0 or int(h["NAXIS3"]) != len(modes):
        raise ValueError("the data set must cover all modes of the mode cube, starting at 0")

    resp = fits.getdata(f"{prefix}_{sci}_resp.fits").astype(np.float32)
    even = fits.getdata(f"{prefix}_{sci}_even.fits").astype(np.float32)
    ref = fits.getdata(f"{prefix}_{sci}_ref.fits")
    mref = ref.mean(0, dtype=np.float64)
    darkf = f"{prefix}_{sci}_dark.fits"
    if os.path.exists(darkf):
        mref -= fits.getdata(darkf).astype(np.float64)
    else:
        mref -= np.median(np.concatenate([mref[:30].ravel(), mref[-30:].ravel()]))

    py, px = np.unravel_index(np.argmax(mref), mref.shape)
    win = np.clip(mref[py - 3:py + 4, px - 3:px + 4], 0, None)
    wy, wx = np.mgrid[-3:4, -3:4]
    c = np.array([px + (win * wx).sum() / win.sum(), py + (win * wy).sum() / win.sum()])

    yy, xx = np.mgrid[0:mref.shape[0], 0:mref.shape[1]]
    far = np.hypot(xx - c[0], yy - c[1]) > 0.42 * min(mref.shape)
    rf = resp[:, far]
    sig = 1.4826 * np.median(np.abs(rf - np.median(rf, 1)[:, None]), 1)
    ef = even[:, far]
    esig = 1.4826 * np.median(np.abs(ef - np.median(ef, 1)[:, None]), 1)

    return dict(resp=resp, even=even, mref=mref, c=c, amp=poke_amplitudes(prefix, modes, sci), sig=sig,
                esig=esig, yy=yy, xx=xx, head=h)


def geometry(d, n, nmin=4, nmax=20, search=4, clip=2.5, fit_center=False, M0=None):
    """Fit the speckle geometry M (px per cycle) from the even cube, normalized to a positive trace.

    A first fit uses each mode's brightest off-core pixel (sp.fit_geometry), with the SNR cut lowered
    to the 70th percentile (at least 5) when fewer than 100 modes pass the default cut of 20. It is then
    refined robustly: for both speckles of each mode with nmin < |n| < nmax, the peak of the 3x3-smoothed
    even image within +-search px of the prediction, then least squares with points beyond clip px
    rejected. Returns M, the rms residual of the kept points [px], and the number kept.

    fit_center: also fit the optical-axis position and update d["c"] (needed with a focal-plane mask,
    where the ref maximum is not the axis). Each mode then contributes only its brighter speckle, so
    it also works when a knife edge blocks one of the pair. M0: the starting M, e.g. from a data set
    without the mask; otherwise it comes from the brightest off-core pixels.
    """
    from scipy.ndimage import uniform_filter

    even, c = d["even"], d["c"]
    pk = sp.even_peaks(even, c[0], c[1], core=8)
    snr = pk[:, 2] / pk[:, 3]
    smin = 20 if (snr > 20).sum() >= 100 else max(5.0, np.percentile(snr, 70))
    if M0 is None:
        M, _, _ = sp.fit_geometry(n, pk, snr_min=smin)
        if np.trace(M) < 0:
            M = -M
    else:
        M = np.array(M0, float)

    fr = np.hypot(*n.T)
    ny, nx = even.shape[1:]
    def collect(M, c):
        X, Y = [], []
        for k in np.where((fr > nmin) & (fr < nmax))[0]:
            e = uniform_filter(np.asarray(even[k], dtype=np.float64), 3)
            cand = []
            for s in (1, -1):
                q = c + s * (M @ n[k])
                x0, y0 = int(round(q[0])), int(round(q[1]))
                if not (search + 2 <= x0 < nx - search - 2 and search + 2 <= y0 < ny - search - 2):
                    continue
                sub = e[y0 - search:y0 + search + 1, x0 - search:x0 + search + 1]
                dy, dx = np.unravel_index(np.argmax(sub), sub.shape)
                cand.append((sub[dy, dx], s * n[k], [x0 - search + dx, y0 - search + dy]))
            if fit_center and cand:
                cand = [max(cand, key=lambda v: v[0])]
            for _, xn, yp in cand:
                X.append(xn)
                Y.append(yp)
        return np.array(X), np.array(Y, float)

    def solve(X, Y, keep):
        if fit_center:
            A = np.c_[X, np.ones(len(X))]
            sol = np.linalg.lstsq(A[keep], Y[keep], rcond=None)[0]
            return sol[:2].T, sol[2]
        return np.linalg.lstsq(X[keep], Y[keep] - c, rcond=None)[0].T, c

    for _ in range(2 if fit_center else 1):
        X, Y = collect(M, c)
        keep = np.ones(len(X), bool)
        for _ in range(4):
            M, c = solve(X, Y, keep)
            r = np.hypot(*(Y - c - X @ M.T).T)
            keep = r < clip
    M, c = solve(X, Y, keep)
    r = np.hypot(*(Y - c - X @ M.T).T)
    if fit_center:
        d["c"] = np.asarray(c, float)
    return M, float(np.sqrt(np.mean(r[keep] ** 2))), int(keep.sum())


def solve_Z(o1, o2, th1, th2, s):
    """Per-pixel Z = sqrt(2) kappa C* A from a cos/sin pair's odd parts per um; s = +1 or -1 (see module doc)."""
    Mx = np.array([[-s * np.sin(th1), -np.cos(th1)], [-s * np.sin(th2), -np.cos(th2)]])
    Mi = np.linalg.inv(Mx)
    return (Mi[0, 0] * o1 + Mi[0, 1] * o2) + 1j * (Mi[1, 0] * o1 + Mi[1, 1] * o2)


def load_modes(path):
    return fits.getdata(path).astype(np.float64)
