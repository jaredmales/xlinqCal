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
    """Load resp, even, mean bias-subtracted ref, PSF center, amplitudes, and per-mode resp noise."""
    h = fits.getheader(f"{prefix}_{sci}_even.fits")
    if int(h["POKE0"]) != 0 or int(h["NAXIS3"]) != len(modes):
        raise ValueError("the data set must cover all modes of the mode cube, starting at 0")

    resp = fits.getdata(f"{prefix}_{sci}_resp.fits").astype(np.float32)
    even = fits.getdata(f"{prefix}_{sci}_even.fits").astype(np.float32)
    ref = fits.getdata(f"{prefix}_{sci}_ref.fits")
    mref = ref.mean(0, dtype=np.float64)
    mref -= np.median(np.concatenate([mref[:30].ravel(), mref[-30:].ravel()]))

    py, px = np.unravel_index(np.argmax(mref), mref.shape)
    win = np.clip(mref[py - 3:py + 4, px - 3:px + 4], 0, None)
    wy, wx = np.mgrid[-3:4, -3:4]
    c = np.array([px + (win * wx).sum() / win.sum(), py + (win * wy).sum() / win.sum()])

    yy, xx = np.mgrid[0:mref.shape[0], 0:mref.shape[1]]
    far = np.hypot(xx - c[0], yy - c[1]) > 0.42 * min(mref.shape)
    rf = resp[:, far]
    sig = 1.4826 * np.median(np.abs(rf - np.median(rf, 1)[:, None]), 1)

    return dict(resp=resp, even=even, mref=mref, c=c, amp=poke_amplitudes(prefix, modes, sci), sig=sig,
                yy=yy, xx=xx, head=h)


def geometry(d, n):
    """Fit the speckle geometry M (px per cycle) from the even cube, normalized to a positive trace."""
    pk = sp.even_peaks(d["even"], d["c"][0], d["c"][1], core=8)
    M, res, ngood = sp.fit_geometry(n, pk)
    if np.trace(M) < 0:
        M = -M
    return M, res, ngood


def solve_Z(o1, o2, th1, th2, s):
    """Per-pixel Z = sqrt(2) kappa C* A from a cos/sin pair's odd parts per um; s = +1 or -1 (see module doc)."""
    Mx = np.array([[-s * np.sin(th1), -np.cos(th1)], [-s * np.sin(th2), -np.cos(th2)]])
    Mi = np.linalg.inv(Mx)
    return (Mi[0, 0] * o1 + Mi[0, 1] * o2) + 1j * (Mi[1, 0] * o1 + Mi[1, 1] * o2)


def load_modes(path):
    return fits.getdata(path).astype(np.float64)
