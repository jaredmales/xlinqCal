# LinQ with xlinqCal calibrations: analysis and implementation assessment

*2026-11-19. Analysis of `planning/linq.tex` for a deterministic 2400-mode ± poke calibration, with $p = 10800$ active WFS pixels and $q = 300\times300 = 90000$ science pixels. Also references the public LinQ code (`Sand-jrd/WFS2PSF-LinQmodel`, commit `e2d865d`).*

*Sections 1–8 are the original analysis. **§9 revises it after the User Clarifications** (the goal is $|S|^2$, not $I$). Where they conflict, §9 wins.*

## Summary

1. **LinQ as written cannot run at full dimension.** Keeping all modes ($n = 2400$) gives $d_\Phi = 2{,}883{,}600$ features. The regression matrix $\mathbf A$ would then be about 1 TB (float), and $\Phi^\top\Phi$ about 33 TB. The reference code forms $\Phi$ and $\mathbf A$ explicitly, so it is limited to roughly $n \lesssim 300$ (§1).
2. **PCA adds nothing on this data, and truncating it is harmful.** On deterministic ± pokes, PCA of the WFS data is just the SVD of the WFS interaction matrix $\mathbf R_w$. The variance spectrum reflects the poke basis and amplitude, not any natural statistics. Truncating to a few hundred components would discard modes that each produce their own science-camera speckles. The right reduced coordinates are the modal coefficients from the calibration's own reconstructor, $z = \mathbf R_w^{+}(x - x_0)$, truncated only for poorly sensed modes (§2).
3. **The current xlinqCal products only contain the linear model.** `(plus − minus)/2` is the odd part, which is linear plus cubic. The quadratic information is in the even part, `(plus + minus)/2 − I0`, and xlinqCal currently discards it. **xlinqCal must save both sign means and an interleaved unpoked reference** (§5.1).
4. **Single-mode pokes can never determine cross terms.** Even with the even part, single-mode pokes give only the linear terms and the diagonal quadratic terms $Q_{kk}$. Cross terms $Q_{kl}$, $k\neq l$, are the interference between modes, which dominates coherent speckle. They are unobservable from single-mode pokes, and no regression method changes that. On this data, LinQ's regression reduces to finite differences (§2.3).
5. **The physics gives the quadratic term a much smaller structure than LinQ assumes.** At each science pixel, $I = |E_0 + \mathbf g^\top c|^2 + I_{\rm inc}$, so the quadratic form is (approximately) **rank-2 PSD per pixel**. In a **Fourier mode basis** it is also **local**: only about $2\pi r^2$ modes (with $r$ in λ/D) affect a given pixel. Either property shrinks the cross-term problem from $2.9\times10^6$ unknowns per pixel to $10^2$–$10^3$ (§3).
6. **Recommended path:**
   - **A, linear model:** possible now with existing data.
   - **B, diagonal quadratic:** needs the xlinqCal change.
   - **C/D, cross terms:** from about 10³ random multi-mode ± probe pairs, using either local LinQ (per-patch quadratic regression) or a low-rank/field model.
   - **E, global kernel-ridge LinQ:** this is exactly the ridge formula in `linq.tex`, but it never forms $\Phi$ or $\mathbf A$. Use it as a full-dimension baseline.

   All of these fit on one workstation, optionally with a GPU (§4, §7).
7. **Test data on dynamic sequences needs per-frame WFS features.** A science frame integrates $I(c(t))$ over 250 ms. Its quadratic part depends on $\overline{zz^\top} = \bar z\bar z^\top + \mathrm{Cov}_t(z)$, not just $\bar z$. Averaging camwfs over the exposure, as xlinqCal does, is fine for static pokes. It is wrong for turbulence or on-sky test data, which need per-frame WFS data or real-time moment accumulation (§5.3).

## 1. Dimensions and the cost of LinQ as written

$d_\Phi(n) = n + n(n+1)/2$ features per science pixel. $\mathbf A$ is $d_\Phi \times q$. The training feature matrix is $N \times d_\Phi$.

| $n$ (WFS components) | $d_\Phi$ | $\mathbf A$ (float) | $\Phi$, $N=4800$ | $\Phi^\top\Phi$ (primal path) |
|---:|---:|---:|---:|---:|
| 100 | 5,150 | 1.9 GB | 0.1 GB | 0.1 GB |
| 300 | 45,450 | 16 GB | 0.9 GB | 8 GB |
| 500 | 125,750 | 45 GB | 2.4 GB | 63 GB |
| 1000 | 501,500 | 181 GB | 9.6 GB | 1 TB |
| **2400** | **2,883,600** | **1.04 TB** | **55 GB** | **33 TB** |

How the reference code (`wfs2psf_fit.c`) works:
- **Explicit matrices:** it expands $\Phi$ explicitly (`quadratic_expand_*`), solves the dual (`N < z_dim`, $\Phi\Phi^\top$ with `posv`) or primal ($\Phi^\top\Phi$, also used in chunked mode) system, and stores $\mathbf B = \mathbf A$ as `z_dim × target_dim`.
- **Science-side reduction:** it shrinks `target_dim` with per-patch PCA of the science images (`-ny`, `-patchsize`).

So at $n=2400$ it is infeasible in memory, whatever is done on the science side. Practical $n$ is a few hundred.

Possible but not useful: about $n = 100$–$300$ global components. As §2 explains, this throws away most of the modes that drive the science image.

## 2. What the deterministic calibration measures

### 2.1 Model
Write the WFS signal and the science image as functions of the modal coefficient vector $c$ (2400 modes), expanded to second order:

$$x(c) = x_0 + \mathbf R_w c + \tfrac12 \mathcal Q_w(c,c) + \dots,\qquad y(c) = y_0 + \mathbf L c + \mathcal Q(c,c) + \dots$$

where, per science pixel, $\mathcal Q(c,c) = c^\top \mathbf Q(\text{pixel})\, c$.

For a single-mode poke $c = \pm a\,e_k$:

- **odd part:** $\tfrac12[y(+a e_k) - y(-a e_k)] = a\,\mathbf L_{:,k} + O(a^3)$. This is what xlinqCal saves now.
- **even part:** $\tfrac12[y(+a e_k) + y(-a e_k)] - y_0 = a^2 \mathbf Q_{kk} + O(a^4)$. This is currently discarded, and $y_0$ is not measured.

The same holds for the WFS: the odd part gives $\mathbf R_w$, and the even part gives the WFS's own quadratic response.

### 2.2 PCA on this data
Centered training rows are $\pm a\,\mathbf R_{w,:,k}$ plus a common even part. PCA is therefore (up to the even-part spread) the SVD of $\mathbf R_w$:
- **The singular spectrum is set by $a$ and the conditioning of $\mathbf R_w$**, not by the statistics of the science data.
- **Truncating by variance amounts to a truncated-SVD reconstructor.** That is reasonable regularization for poorly sensed modes, but it is not a large dimension reduction. Every well-sensed mode makes its own speckles, so the rank stays near $\min(2400, N_{\rm act})$.
- **Better to construct the coordinates explicitly:** $z = \mathbf R_w^{+}(x - x_0)$, with SVD truncation chosen from the WFS noise. $\mathbf R_w$ is 10800 × 2400 (104 MB float), and its SVD takes about $4\times10^{11}$ flops (seconds to a minute with LAPACK).

**Actuator count:** the BMC 2K has about 2040 active actuators. Please confirm, since the mask isn't on this machine. A 2400-mode set is then overcomplete (rank ≤ 2040), and the modal coordinates are only defined up to a ≥360-dimensional null space. Either the basis should have ≤ 2040 modes, or the reconstructor must truncate. Resolution: the 2400 are the square-Nyquist Fourier set, and a truncated reconstructor is fine (§9.3).

### 2.3 What regression can learn from single-mode pokes
In modal coordinates, each training sample's quadratic feature vector is $a^2 e_k e_k^\top$. These span only the 2400 diagonal directions of the 2.9M-dimensional quadratic feature space:
- **Ridge or least squares on this data** reproduces the finite differences: linear = odd/$a$, diagonal = even/$a^2$.
- **The min-norm solution sets every cross term to zero.** Prediction for a multi-mode wavefront becomes the incoherent sum $\sum_k Q_{kk}c_k^2$, which omits the interference terms.

**So single-mode pokes, however many, cannot calibrate LinQ's quadratic term.** Cross terms need training samples with several modes on at once: random multi-mode probes, or natural turbulence/aberration telemetry, which is what the paper trains on.

## 3. The physical structure of the quadratic term

For a coronagraphic (or any) image and small phase, the focal-plane field is linear in $c$:

$$I(\text{pixel}) = |E_0 + \textstyle\sum_k c_k G_k|^2 + I_{\rm inc} = |E_0|^2 + 2\,\mathrm{Re}(E_0^* \mathbf g^\top c) + c^\top \mathrm{Re}(\mathbf g\mathbf g^{H})\,c + I_{\rm inc}$$

where $\mathbf g \in \mathbb C^{2400}$ is the per-pixel complex response (the EFC Jacobian row). Consequences:

- **Rank-2 PSD:** $\mathbf Q = \mathrm{Re}(\mathbf g\mathbf g^H) = \mathbf g_r\mathbf g_r^\top + \mathbf g_i\mathbf g_i^\top$ is **PSD with rank ≤ 2** at each pixel. That is about 4800 real parameters per pixel instead of 2.9M. The second-order field term ($e^{i\varphi}\to -\varphi^2/2$) adds a correction $\propto E_0$. It is small in a dark hole and should be expected outside it.
- **What single pokes give:** odd part $= 2a\,\mathrm{Re}(E_0^* G_k)$, even part $= a^2|G_k|^2$. That determines $|G_k|$ and $\cos\theta_k$ relative to $E_0$, but leaves a **sign ambiguity in $\sin\theta_k$ per mode per pixel**. The relative phases are undetermined where $E_0 \to 0$. Resolving them needs multi-mode probes, or an optical model of $G$.
- **Locality:** a Fourier mode at spatial frequency $f$ (cycles/pupil) makes a speckle pair at $\pm f$ λ/D, about 1 λ/D wide. So at focal position $\xi$, the modes with $|f \mp \xi| \lesssim r$ dominate. That is $m \approx 2\pi r^2$ modes (sin and cos):

  | $r$ (λ/D) | local modes $m$ | cross terms $m(m-1)/2$ | params per pixel (lin + quad) |
  |---:|---:|---:|---:|
  | 1.5 | 14 | 91 | ~120 |
  | 2 | 25 | 300 | ~350 |
  | 3 | 57 | 1,600 | ~1,700 |
  | 4 | 100 | 5,000 | ~5,150 |

  Airy sidelobes fall off slowly, so $r$ trades truncation error against data and compute. A coronagraph or apodizer changes the effective PSF, and $r$ should be chosen from simulation. **Locality holds only in a Fourier-like basis.** Actuator or Hadamard pokes spread over the whole focal plane. The linear part rotates exactly between bases, but measured diagonal quadratic terms do not, so the calibration basis itself should be Fourier modes.
- **Operating point:** the linear term depends on $E_0$, which drifts with NCPA and the coronagraph state, while $|G_k|^2$ does not. A field-form model can be updated by re-estimating only $E_0$ (2 parameters per pixel). A general quadratic regression has to be retrained.

## 4. Approaches, data needs, and costs

Assumptions: $n = 2400$ modes, $p = 10800$, $q = 90000$, float storage. The patch count assumes about 3.5 science pixels per λ/D (≈ 7300 one-λ/D patches in 300×300). Please correct the pixel scale.

| | Model | Calibration data | Fit compute | Model size | Predict / frame |
|---|---|---|---|---|---|
| **A** | Linear: $y = y_0 + \mathbf L z$, $z = \mathbf R_w^+(x-x_0)$ | **Existing** single-mode odd parts | SVD of $\mathbf R_w$, ~$4\times10^{11}$ flops | $\mathbf R_w^+$ 104 MB + $\mathbf L$ 864 MB | $2.4\times10^8$ MAC |
| **B** | A + diagonal quadratic $\mathbf D\,(z\circ z)$ | Single-mode even parts + $y_0$ (xlinqCal change) | trivial | + 864 MB | $4.6\times10^8$ MAC |
| **C** | **Local LinQ:** per patch, quadratic regression on local modes ($m$), linear and diagonal terms fixed from A/B | Random multi-mode ± probe pairs, $N_r \gtrsim 3\times$ cross terms (≈ 1000 for $m=25$; 4000–5000 for $m=57$) | per patch QR $N_r\times m^2/2$: ~$10^{12}$ ($m$=25) to ~$10^{14}$ ($m$=57) flops total | ~0.1–0.6 GB | $q\cdot m^2/2$ ≈ $10^7$–$10^8$ MAC |
| **D** | **Low-rank / field:** fit complex $\mathbf g$ (and $E_0$, $I_{\rm inc}$) per pixel, $\mathbf Q$ = rank-2 PSD; local or global | Single-mode odd + even, plus $N_r$ ≈ 4–6 × (2m) random ± probes (a few hundred local) to fix relative phases | per pixel alternating least squares / gradient, ~$10^{11}$–$10^{12}$ flops total | 1.7 GB global (complex $\mathbf g$, 2400 modes) | $2\times2.2\times10^8$ MAC |
| **E** | **Global LinQ via kernel ridge.** Exactly the `linq.tex` ridge solution $\mathbf A = \Phi^\top(\Phi\Phi^\top+\lambda I)^{-1}Y$, with $n$=2400 and no truncation | All frames: ±single pokes + random probes, $N$ ≈ 5–15 × 10³ | Gram $K = G + \tfrac12(G\circ G + H)$, $G = ZZ^\top$, $H = (Z\circ Z)(Z\circ Z)^\top$: ~$10^{12}$; Cholesky + solve with $q$ RHS: ~$10^{13}$ | $\alpha$: $N\times q$ = 1.7–5.4 GB | $N(n+q)$ ≈ $10^9$ MAC (~0.1–1 s CPU) |

Notes:
- **E** reproduces LinQ's full quadratic regression with nothing truncated. The kernel identity holds for the $i\le j$ feature convention: $\phi(z)\cdot\phi(w) = z\cdot w + \tfrac12[(z\cdot w)^2 + \sum_i z_i^2 w_i^2]$. Its generalization of the cross terms is limited by data, not memory. Without a locality or low-rank prior, $10^4$ samples cannot pin down $2.9\times10^6$ cross-term directions. It is a useful baseline and a check that C/D improve on it.
- **C** is LinQ restricted by physics: the same regression, done per patch with features from local modes only. Patches share a design matrix, so each patch needs one factorization with many right-hand sides. Speckles can be predicted only within the DM control radius. Outside it, use A/B, or include that region in the fit and accept a poorer model.
- **D** uses the most structure and needs the fewest probes. This is the PhaseLift/low-rank setting: $O(\text{rank}\cdot m)$ quadratic measurements recover a rank-2 PSD matrix. It is nonconvex per pixel; initialize from the single-mode $|G_k|$ and $\cos\theta_k$, or from an optical model. A cheaper variant: fit **C**, then project each $\mathbf Q$ onto rank-2 PSD.
- Restrict the fit to a region-of-interest mask (e.g. the dark hole and control region). That scales $q$ and every cost in the table linearly.

## 5. Changes needed to calibration and test data

### 5.1 xlinqCal (small)
- **Save sign means, not only the difference.** Per poke, write the camsci1 and camwfs means for + and − (or the even part directly). The difference cube is then derived. Extra output: 864 MB per sci cube and 138 MB per WFS cube (full 120×120 frame).
- **Unpoked reference $y_0$, $x_0$:** interleave zero-poke acquisitions, e.g. `0 + - - + 0` per poke, or a 0 every M pokes with interpolation. The even part is a small difference of large images, so $y_0$ must be measured close in time. With `+--+` the sign-mean windows are symmetric about the poke's center time, so a reference at that center time, or the mean of references before and after, cancels linear drift. Darks cancel in the even part as they do in the odd part.
- **Amplitude:** phase per mode is $\varphi = 4\pi a/\lambda$ (surface $a$, reflection). At 656 nm, $a$ = 10 nm rms gives 0.19 rad. The even part has a quartic contamination of relative size $O(\varphi^2)$, and the odd part a cubic one. Optionally take two amplitudes ($a$, $2a$) and extrapolate: $Q \approx [16E(a) - E(2a)]/(12a^2)$. The SNR of the even part scales as $a^2$, so amplitude needs a measured trade-off on camsci1.
- **Basis:** use Fourier (sin/cos) modes for locality (§3), with ≤ N_act modes or explicit truncation.

### 5.2 Random multi-mode ± probes (no new code)
xlinqCal already supports this: a poke file of random multi-mode shapes, with each shape normalized to the total probe rms, is just another calibration. With the 5.1 change it also yields the even parts $c_r^\top\mathbf Q c_r$, which directly measure cross terms with drift cancelled. Keep the total probe phase small enough for the quadratic model (≲ 0.1–0.3 rad rms, to be tuned). Time: with `nRepeat=1`, `nFrames=2`, a probe pair takes about 8 camsci1 frames (2 s at 4 Hz). So 1000 pairs ≈ 35 min, 2000 ≈ 67 min, 5000 ≈ 2.8 h. For comparison, the full single-mode set is about 7.7 h with the default settings (46 frames per poke). Reducing `nFrames`/`nRepeat` for the single-mode set trades that time for SNR.

### 5.3 Test sequences (dynamic) and WFS averaging
For time-varying wavefronts, the training/prediction feature for one science frame must be $\overline{\phi(z)} = [\bar z,\ \overline{z_i z_j}]$, averaged over **per-frame** WFS measurements during the exposure. Averaging $x$ first and then expanding drops $\mathrm{Cov}_t(z)$, which for AO residuals is most of the halo. Options:
- **Record full-rate camwfs** (`camwfs-sw`) synchronized with camsci1: about 2 kHz × 10800 px × 2 B ≈ 43 MB/s, or 155 GB/h. Expand offline.
- **Accumulate moments in real time** in an xlinqCal-like recorder: $\sum_t z_t$ and $\sum_t z_t z_t^\top$ per science frame. The full matrix is $2.9\times10^6$ MAC per frame, or $5.8\times10^9$ MAC/s at 2 kHz (fine on a GPU). Local pairs only ($m$ ≈ 50) need about $2.4\times10^8$ MAC/s, which a CPU handles.

Also correct the noise bias: $\overline{z_iz_j}$ includes the WFS noise covariance. Subtract it, or estimate it from interleaved halves.

## 6. Notes on LinQ as described in `linq.tex`

- **Intercept:** the text says no constant is needed because inputs are centered. But quadratic features of centered inputs are not centered ($E[z_iz_j]\neq0$). The reference code does append a constant column (`Z[i*z_dim + z_dim-1] = 1.0`). Keep it, or center $\Phi$.
- **Automatic λ is a numerical floor, not regularization.** $\lambda = \text{reg}\cdot\epsilon\, d_\Phi\,\lambda_{\max}$ is about $6\times10^{-10}\lambda_{\max}$ at $n$=2400 in double precision. In the underdetermined case that is the min-norm interpolant, which fits noise exactly. Choose λ by holdout or GCV instead. GCV is cheap from an eigendecomposition of the $N\times N$ Gram matrix. Also, the text computes $\lambda_{\max}$ from $Z^\top Z$ (linear features), not from $\Phi$; the code uses $\Phi$.
- **One λ for linear and quadratic features:** the two blocks scale as $a$ and $a^2$, so penalizing them with the same λ weights them arbitrarily. Use separate weights (a two-parameter kernel $K = \beta_1 G + \beta_2 K_{\rm quad}$ in E, or block scaling in C).
- **Science-side PCA:** the code's per-patch PCA reduces the target dimension. On deterministic pokes, though, the science image space spanned by the responses is also basis-driven (rank ≤ 2400 + diagonal terms). It doesn't help with the real bottleneck, $d_\Phi$.

## 7. Implementation plan

Offline tooling. The fitting uses large dense linear algebra. Use C++ with mxlib/Eigen and LAPACK, with cuBLAS/cuSOLVER (available through mxlib's CUDA support) for E and C at large $m$. Or prototype in Python/NumPy first, which is faster for algorithm comparison, and port the chosen method.

1. **Simulation before instrument time.** Extend the test simulator, or write a separate Python model, to a Fourier-optics model: 50×50 DM with Fourier modes, pupil, optional coronagraph, $|{\rm FFT}|^2$ science image, and a linearized PyWFS (or a simple WFS model with a configurable quadratic term). Generate single-mode ± sets, random probe sets, and dynamic test sequences. Use it to compare A–E and choose $r$, the probe amplitude, and $N_r$.
2. **xlinqCal changes (§5.1):** save sign means, interleaved references, and optionally two amplitudes. The simulator integration test is extended to check the even part against a known quadratic.
3. **A (linear) on existing data:** `linqFit` builds $\mathbf R_w^+$ and $\mathbf L$; `linqPredict` applies them. Validate on random probe data. A held-out single-mode poke is not a real test, because it isn't in the span of the remaining training data.
4. **B (diagonal)** once 2 is available.
5. **E (global kernel ridge)** as the full-dimension LinQ baseline. N ≈ 10⁴ needs about 1 GB of Gram matrix and 3.6 GB of $\alpha$, and fits in minutes.
6. **C (local LinQ)**, then **D (low-rank/field)**. Compare on held-out random probes and on a dynamic test sequence (§5.3).

Resource summary for the largest step (C with $m$ = 57, or E with N = 1.5 × 10⁴): ≤ 10¹⁴ flops to fit (hours on CPU, minutes on GPU) and ≤ 10 GB of memory. Prediction for an hour of camsci1 at 4 Hz (14,400 frames) costs ≤ $2\times10^{13}$ MAC.

## 8. Questions

1. What is the 2400-mode basis, and on how many active actuators? (≈2040 for the BMC 2K implies an overcomplete set.) Would you switch to a Fourier basis for the LinQ calibration?

2400 is the number of Fourier modes made by 48 actuator diameter mirror. The modal basis could be those Fourier modes, but I was planning to use a KL basis for a von Karmen PSD constructed from the Fourier modes. This already exists.  

2. What are the camsci1 plate scale and wavelength (pixels per λ/D), and the coronagraph configuration? Is there a dark-hole region of interest to restrict $q$?

Assume 4.8 pix/ (lambda/D).  I chose 300x300 in the details to enclose the 48x48 lambda/D contorl region of the DM.

3. What will the test data be: random probes, turbulence injected on the internal source, or on-sky? Can full-rate camwfs be recorded synchronized with camsci1 for those sequences?

For testing we can use random probes, we'll produce them from model-AO-filtered phase screens. Yes we can record synchro data.

4. Is the even-part calibration worth the extra time (references plus possibly two amplitudes), and what total calibration time is acceptable?

Perhaps.  See below for expanded discussion.

5. How would you like to compare against the reference implementation? It runs as-is at $n$ ≈ 100–300 on $z$ (instead of its own PCA), which makes it a useful sanity check on small problems.

I think we can run the reference implementation on the test data.  We train on part and test on part.

# User Clarifications

My goal here is to predict only the difference between the image and the reference PSF.  I'm basing the calibration procedure on:

I = |C|^2 + |S|^2 + 2*Real(C* x S)

Where C is the deterministic field and S is the random field (the speckles).  The goal is |S|^2, not I. 

What I want to test in the end is if |S|^2 can be used to improve our usual focal plane only post-processing methods, by including frames of |S|^2 as part of the basis set.  This is motivated by the fact that C is variable, and can change with different coronagraph configurations and dark hole settings, but the mapping from WFS to |S|^2 should be fairly stable.

The calibration procedure was constructed to get this mapping.  Then in practice when we are recording varying wavefronts, we have to reconstruct the sum(|S|^2_wfs) over the science exposure. 

## 9. Revised assessment after the User Clarifications (2026-11-19)

### 9.1 How the ± calibration maps onto $I = |C|^2 + |S|^2 + 2\,\mathrm{Re}(C^*S)$
With $S = \sum_k c_k G_k$ (field linear in the modal coefficients), a ± poke separates the three terms exactly, to this order:

| Product | Content | Depends on $C$? |
|---|---|---|
| $I(0)$ (reference PSF) | $\lvert C\rvert^2$ | yes |
| odd part $\tfrac12[I(+c)-I(-c)]$ | $2\,\mathrm{Re}(C^*S)$ | **yes**: the pinned-speckle cross term |
| even part $\tfrac12[I(+c)+I(-c)]-I(0)$ | $\lvert S\rvert^2$ | **no**: the target |

At second order the even part also contains the C-dependent dimming term $2\,\mathrm{Re}(C^*S_2)$; it was measured in try1 (§9.10).

Consequences:
- **What xlinqCal saves now is the C-dependent term.** The current product, the odd part, is exactly the piece you expect to change between coronagraph and dark-hole configurations. On the camsci1 side it isn't the |S|² mapping. On the camwfs side it remains essential: it gives $\mathbf R_w$ and the reconstructor $z = \mathbf R_w^+(x-x_0)$.
- **The even part is required, not optional.** For your goal, the §5.1 change (save both sign means plus an interleaved $I(0)$) is the core of the calibration. The single-mode even parts are $|G_k|^2 = |S_k|^2$, one C-independent speckle image per mode. That is the diagonal of the |S|² quadratic form.
- **A random ± probe directly measures |S|².** For any multi-mode probe $c_r$, the even part is $|S_r|^2 = |\mathbf G c_r|^2$, with $C$ cancelled, plus quartic terms. So ± pairs of your model-AO-filtered phase screens give **training and test pairs (WFS, |S|²) with ground truth**. This is LinQ's native regime (random samples), so a learned regression is well posed there, unlike on single-mode pokes. xlinqCal handles it already (screens as the poke file), apart from the even-part change.
- **Bonus product:** the odd parts give $2\,\mathrm{Re}(C^*S)$ for the current configuration. In a science frame its mean is about $2\,\mathrm{Re}(C^*\mathbf G\bar z)$, linear in the mean WFS. That could serve as an extra, configuration-specific basis component if the pinned term matters (§9.5).

### 9.2 The model for |S|²
$|S|^2$ is purely quadratic in $c$, with no linear term and no intercept. Per pixel it is $|\mathbf g^\top c|^2$, a **rank-2 PSD** quadratic form (§3). That gives three levels of structure:

1. **Learn the complex Jacobian $\mathbf G$** (q × n complex, 1.7 GB) from intensity data. This is per-pixel phase retrieval, a PhaseLift-type problem. The per-pixel global phase is unobservable but **irrelevant for |S|²**, which removes the sign/phase-reference problem of §3. Unknowns per pixel: $2m-1$. Data: $m$ single-mode even parts (|G_k|², an initialization and constraint) plus $N_r$ probe even parts, with $N_r \gtrsim 4$–$8\times m$. This is the most data-efficient approach and generalizes beyond the training distribution. Fit by per-pixel alternating least squares or Wirtinger-flow gradient steps; it is embarrassingly parallel over pixels.
2. **Quadratic regression** (local LinQ, §4 C, or global kernel ridge, §4 E) with **quadratic features only**. Drop the linear block and the intercept. That also removes the λ-weighting issue of §6. Fewer assumptions, more data.
3. **Calibrated optical model:** $S \approx \mathcal F\{P\, i\varphi\}$ through the coronagraph, with geometric parameters (plate scale, rotation, distortion, pupil and Lyot) fitted to the measured $|G_k|^2$. Compute $S_t$ by FFT of the reconstructed phase. This is a useful prior or initialization for (1), and a physics check.

Reference LinQ (train/test split on the probe data) is a good baseline, as you suggested. Use $z$ (or its own PCA) as input and the even-part |S|² as target. At $n \le 300$ it will mainly capture speckles near the center.

### 9.3 Basis: Fourier vs KL
For the calibration I recommend the **Fourier basis**, even if KL is used elsewhere:
- **Locality.** A Fourier mode's |S_k|² is a speckle pair at $\pm f$, so each pixel couples to only $m \approx 2\pi r^2$ modes (r in λ/D): about 25 at r = 2, 57 at r = 3. KL modes for an isotropic von Kármán PSD are nearly degenerate across orientation at fixed |f|, so they mix Fourier modes around a ring. Their speckles spread in azimuth, and each pixel then couples to about $2\pi|\xi|\cdot 2r$ modes (≈ 150 at $|\xi|$ = 12 λ/D, r = 2, versus ≈ 25 for Fourier). That costs roughly 6× in probe count (approach 1) or about 40× in cross-term unknowns (approach 2).
- **No information is lost.** KL is a fixed linear transform $T$ of the Fourier modes, so $z_{\rm KL} = T^{-1} z_{\rm F}$ and $\mathbf G_{\rm KL} = \mathbf G_{\rm F} T$ are exact. Only the measured diagonals $|G_k|^2$ are basis-specific, and they are more useful in the Fourier basis.
- **Overcomplete set.** The 2400 modes are the $(2\cdot24+1)^2 - 1$ frequency points of the square Nyquist region $|u|,|v| \le 24$ cycles/pupil (the user's clarification). (try1 showed the modes are cycles per 50-actuator grid; see §9.10.) On the actuator grid, the $\pm24$ edges are degenerate: they alias onto each other, and sin vanishes at Nyquist. Only the illuminated actuators (≲ 2040 active) carry signal. So the set is linearly dependent, and $\mathbf R_w$ has a numerical null space. That's fine if $z$ comes from a truncated pseudo-inverse and the same $z$ convention is used to fit and apply $\mathbf G$. $S = \mathbf G z$ is then unique even though $z$ is not.

Is there a reason the calibration itself needs KL (e.g. SNR ordering)? If so, a KL calibration still works with approaches 1–3, at the cost above.

### 9.4 Sizes with 4.8 px/(λ/D)
- 300 px ≈ 62.5 λ/D across. The control region is the **square** $|\xi_x|,|\xi_y| \le 24$ λ/D, about 230 × 230 px. Its corners, at 34 λ/D radius, sit at (±115, ±115) px, inside the 300 × 300 frame with about 35 px margin for speckle halos.
- A pixel mask restricted to the square control region has $q \approx (48\times4.8)^2 \approx 53{,}000$ px, versus 90,000. That cuts every science-side cost and size in §4 by about 1.7×. Keep the full frame only if out-of-region speckles (sidelobes, aliasing) are wanted.
- One-λ/D patches: 2,304 inside the control region, or about 3,900 for the full frame (not 7,300).

### 9.5 Applying it to varying wavefronts
$\sum_t |S_t|^2 = \sum_t |\mathbf G z_t|^2$ needs per-WFS-frame $z_t$. Use the synchronized full-rate camwfs recording, offline:

| Method | Cost per WFS frame (q = 53k / 90k) | 1 h at 2 kHz (7.2 × 10⁶ frames) |
|---|---|---|
| Full complex $\mathbf G$, $S_t = \mathbf G z_t$ | 2qn ≈ 2.5 / 4.3 × 10⁸ MAC | 1.8 / 3 × 10¹⁵ MAC: GPU, minutes |
| Local $\mathbf G$ ($m$ = 57 per pixel) | 2qm ≈ 6 / 10 × 10⁶ MAC | 4 / 7 × 10¹³ MAC: CPU, ~hour |
| Optical model (FFT, ~512²) | ~5 × 10⁷ flop | ~4 × 10¹⁴ flop: GPU, minutes |
| Quadratic regression | form $\sum_t z_tz_t^\top$ (local pairs), then apply | similar to the local $\mathbf G$ row |

The pinned term per frame is $2\,\mathrm{Re}(C^*\mathbf G\bar z)$, from the odd-part calibration for that configuration. Whether to include it as another basis frame is a post-processing choice. It is linear in $\bar z$, so it is cheap.

### 9.6 Assumptions to test
- **|S|² independent of configuration.** $C$ changes with FPM and dark-hole DM settings, but $\mathbf G$ also depends on the Lyot stop and apodizer, and weakly on the dark-hole DM shape, through $e^{i\varphi_{\rm DH}}$. Test directly by comparing single-mode even parts $|G_k|^2$ for a subset of modes in two configurations.
- **Quadratic regime of the probes.** AO-filtered screens put most variance at low order. Large low-order amplitudes make the field nonlinear near the core (quartic contamination of the even part ~$\sigma^2$). Small high-order amplitudes give low SNR for the |S|² of high-order modes, far from the core. To train $\mathbf G$ (approach 1), a flatter probe PSD at total phase ≲ 0.1–0.3 rad rms is better. To test realism, use screens with the target statistics. These can be two probe sets.
- **WFS nonlinearity.** Use the odd WFS part $(x_+ - x_-)/2$ to get $z$ for training; it removes the WFS's own quadratic response. At application time $x$ includes that response, so check the bias on the test set.

### 9.7 Revised calibration plan and time (camsci1 at 4 Hz)
1. **Single-mode Fourier set** with sign means and interleaved $I(0)$. This gives $\mathbf R_w$, $|G_k|^2$, and the pinned term for the current configuration. At `nFrames=4`, `nRepeat=1`: about 12 frames per poke plus references, ≈ 2.2 h for 2400 modes. The current defaults take ≈ 7.7 h. Decide on SNR from a short trial.
2. **Training probes:** about 500–1000 ± pairs of flattened-PSD random screens for $\mathbf G$ (local $m$ ≈ 57 → $N_r$ ≈ 250–450 for approach 1; ≈ 1000 for robustness), plus the AO-filtered screens for regression approaches. 8 frames per pair: 500 ≈ 17 min, 1000 ≈ 35 min.
3. **Test probes:** about 500 held-out AO-filtered screens (≈ 17 min), plus synchronized dynamic sequences with full-rate camwfs.

### 9.8 Revised next steps
1. ~~xlinqCal: save sign means and interleaved reference frames (§5.1).~~ Done 2026-11-19: `mean`, `even`, and `ref` cubes, `ref.every`; see `initial_plan.md` §5.7.
2. Simulator: Fourier-optics |S|² model (§7.1), to validate approaches 1–3 and the probe design offline.
3. Fit $\mathbf G$ (approach 1) and local quadratic regression (approach 2) on simulated data. Then run reference LinQ on the same train/test split.

### 9.9 Using calibrated WFS frames (2026-11-19)
Suggestion: use the normalized, reference-subtracted WFS frames instead of raw camwfs. Findings, from cacao `acquireWFSim.c` (milk-xwcl `1286da59`) and `/opt/MagAOX/config`:

- **The tweeter loop is `aol1`.** acquWFS writes `aol1_imWFS0` (dark-subtracted), `aol1_imWFS1` (normalized by total flux over `aol1_wfsmask`, **and multiplied by the mask**), and `aol1_imWFS2` (= imWFS1 − `aol1_wfsrefc`). With `compWFSpub` on, it also copies imWFS2 to **`aol1_imWFS2pub`**, which exists for readers outside the loop: readers of imWFS2 itself take its cache lines from the loop (MagAOX AGENTS rule 28, `shmim-monitor-speedups.md`). imWFS2pub carries the camera `atime`.
- **No code change is needed:** set `wfs.shmim=aol1_imWFS2pub`. It is float, the camwfs frame size, and updates once per camwfs frame, so the gate logic is unchanged. Preflight catches a stale stream, e.g. acquWFS not running or pub off.
- **Benefits:**
  - Source-flux fluctuations are removed from the WFS signal.
  - Pixels outside the mask are zero, which gives the 10800 active pixels directly.
  - $x_0 \approx 0$.
  - Responses are in the same units as the loop's own calibration (`aol1_zrespM`), so they can be compared and reused.
  - **The test/application data path already runs on RTC:** `aol1_imWFS2_cbuff` → `aol1_imWFS2-sw` records full-rate calibrated frames. `aol1_modevalWFSpub` → `aol1_modevalWFS_cbuff` → `aol1_modevalWFS-sw` records the loop's **per-frame modal coefficients** at about 19 MB/s for 2400 modes. That is $z_t$ directly, with no reconstruction step, if the fit is done in the loop's modal basis with the loop's control matrix.
- **The reference must be frozen during a run.** `aol1_wfsrefc` is updated as `mult*(wfsref − wfszpo) + (1−mult)*wfsrefc + gain*imWFS3`, so it changes if `WFSrefcgain` ≠ 0 or another process writes `aol1_wfszpo`. A change inside a `+--+` block corrupts the odd part. A change between references corrupts the even part. User clarification: on MagAO-X `wfsrefc` is never updated and `wfsref` changes only rarely, so no freeze check is implemented. `wfsmask` and the dark must not change during a run either.
- **Normalization is a small nonlinearity:** dividing by the per-frame total flux in the mask adds a term when pokes move light across the mask edge. It appears in the WFS even part and is expected to be negligible. camsci1 is not normalized, so lamp drift there is handled by the reference interpolation.
- **Reference choice doesn't matter at application time.** A different `wfsrefc` between calibration and science shifts $z$ by a constant. Define the random part as $z_t - \bar z$ over the science sequence (or a sliding window), and $\bar z$ goes into $C$. That is consistent with the |C|² + |S|² + 2Re(C*S) decomposition, where $C$ includes the static/mean aberration.
- **Several WFS streams, implemented 2026-11-19:** xlinqCal gates and averages several WFS streams at once (`wfs.shmim=aol1_imWFS2pub,aol1_modevalWFSpub`). That calibrates the response of the loop's modal coefficients alongside the pixel-level response. **Decision (user): use `modevalWFS` as the model input.** The |S|² model ($\mathbf G$, or the quadratic regression) is then fit with $z$ = the `aol1_modevalWFS` coefficients, and applied to sequences recorded by `aol1_modevalWFS-sw`, with no separate reconstruction step. Considerations:
  - **Basis:** $z$ is in the loop's control-mode basis (KL, if that is what aol1 uses). For locality (§9.3), transform to Fourier coordinates with the known KL↔Fourier matrix before fitting. The model is linear in the field, so this is exact.
  - **Calibration of $z$:** the per-poke odd part of `modevalWFS` is the loop's modal response to each poke (≈ amp·e_k if the poke basis equals the control basis). Its even part shows the WFS/reconstructor nonlinearity in modal space.
  - **Truncation:** modes the loop doesn't control are absent from $z$, so the model cannot predict their speckles. The pixel-level `imWFS2pub` products are kept as a fallback and a check.

### 9.10 Findings from the first instrument data set, try1 (2026-11-19)
Data set `try1/xlinqCal_20261119_001`: 2400 Fourier modes, amp 0.05 µm rms, camsci1 at 26.5 Hz, `nFrames=2`, `nRepeat=1`, 18 min. Bookkeeping is in `initial_plan.md` §5.7. Findings that affect this analysis:

- **Mode frequency units.** The modes in `modFourier_48.fits` are integer cycles **per 50 actuators** (`fillFourierBasis(modes, 50, …)`), with $|n_x|, |n_y| \le 24$. Read "cycles/pupil" in §9.3 and "48 λ/D" in §9.4 as cycles per 50-actuator grid. Converting to cycles per pupil needs the illuminated pupil diameter in actuators.
- **Measured camsci1 geometry.** Speckle positions follow pix = M·n with M ≈ [[3.92, 2.03], [−2.03, 3.72]] px per (cycle/50 act). That is 4.43 / 4.27 px per cycle/50 (slightly anisotropic), rotation ≈ −27.5°, det > 0 (no flip), fit residual 0.7 px. `tools/shape_pokes.py` refits it and records it in the shaped cube header (GEOMij).
- **The even part is not pure |S|².** At second order, $e^{i\varphi} \approx 1 + i\varphi - \varphi^2/2$ adds $S_2 \propto -\tfrac12 \mathcal F\{P\varphi^2\}$, and $2\,\mathrm{Re}(C^*S_2)$ is even in the poke. For a single mode it dims the reference by the energy put into the speckles: even $\approx |S|^2 - \varepsilon\,|C|^2$. In try1, the even image outside the speckles fits $-\varepsilon\cdot$ref with $\varepsilon/E$ ≈ 1.7 (IQR 1.4–2.0; E = speckle energy as a fraction of the PSF). The net even flux is ≈ 0.03 E, so energy is conserved. Beyond ~8 λ/D, the dimming term is the dominant even signal (the stacked speckle profile goes negative there). Consequences:
  - Estimate $|S_k|^2 \approx \mathrm{even}_k + \varepsilon_k\,\mathrm{ref}$, with $\varepsilon_k$ fitted per mode, or include the term in the model.
  - The term is C-dependent, so it is part of what changes between configurations (§9.6). With a coronagraph in, $|C|^2$ is small and so is the term.
  - For multi-mode probes (§9.1), $S_2$ also has cross terms at $f_a \pm f_b$, which interfere with $C$. The probe even part is $|\mathbf G c|^2 + 2\,\mathrm{Re}(C^* S_2(c))$, not $|\mathbf G c|^2$ alone. Approach 1 needs this term (or a dark hole) when fitting $\mathbf G$.
- **Amplitude.** At 0.05 µm, low-order speckles carry ~20% of the PSF flux, which is too large for the quadratic regime. Speckle energy falls 67× toward the corners. `tools/shape_pokes.py` makes a shaped cube (target 1% per mode, cap 0.05 µm): 0.010 µm at low order, the cap for $|n| \ge 24$. See `initial_plan.md` §5.7.
- **WFS even/odd ratio** (rms of the even part over rms of the odd part, per mode): ≈ 0.5 for `aol1_imWFS2pub`, ≈ 0.27 for `aol1_modevalWFSpub`. That is a sizable WFS nonlinearity at 0.05 µm, so using the odd part for $z$ (§9.6) matters. In modal space each Fourier poke spreads over many control modes: the largest coefficient carries only ~4%, consistent with a KL control basis (§9.3, §9.9). Modes with $|n| > 28$ respond ~8× more weakly in `modeval`.

**Rejected: poking two or four Fourier modes at once.** The idea was to poke mode pairs $f$ and $R_{90}f = (-v, u)$ (and quads at 45° above some cutoff), whose speckles fall in different camsci1 pixels. The data supported it on the camsci1 side above $|n| \approx 8$–10:
- The worst-case speckle cross term was ≈ 2% of a speckle peak at $|n| = 10$.
- The stacked speckle profile is below $10^{-3}$ of the peak beyond ~5 λ/D.
- It would have saved 44% of pokes with pairs above 10, or 62% with quads above 14.

**Decision (user): rejected, because the WFS responses cannot be split per mode.** In both `imWFS2pub` and `modevalWFS` the two modes overlap, so 2400 per-mode WFS responses need 2400 independent pokes:
- Poking $a+b$ and $a-b$ separates the modes exactly, in both the WFS and the even part, but saves nothing.
- Separating by spatial frequency in the pupil images, or projecting `modeval` onto Fourier modes, assumes a clean WFS response structure that the calibration is meant to measure.
- A separate WFS-only calibration loses the simultaneous WFS and camsci1 measurement.

Time savings must come from amplitude shaping and the choice of `nFrames` and `nRepeat`.
