# xlinqCal

Measures synchronized WFS and camsci1 responses to a sequence of tweeter DM pokes on MagAO-X. The goal is to predict camsci1 images from camwfs images. This is a standalone program: it uses only MILK/ImageStreamIO streams, with no INDI.

## How it works

For each poke shape `k` and each sign in the sequence (`+-`, then `-+`, ... for `seq.nRepeat` pairs; `+--+` by default):

1. Write `initial + sign*amp*shape_k` to `dm.channel` (default `dm01disp04`, written non-passively so `dmcomb` updates even with the loop open). Wait for `dm.comb` (`dm01disp`) to update, and record the projection of its change onto the command as a check.
2. Discard the first camsci1 frame after the poke, then `sci.nSettle` more frames.
3. Average the next `sci.nFrames` camsci1 frames. The frames of each WFS stream between the posts that bound those camsci1 frames are averaged too.

`wfs.shmim` is a comma-separated list of WFS streams, each read by its own thread and gated identically. On MagAO-X, the recommended streams are the cacao tweeter-loop publications for readers outside the loop:
- `aol1_imWFS2pub`: dark-subtracted, flux-normalized, masked, reference-subtracted frames.
- `aol1_modevalWFSpub`: the loop's per-frame modal coefficients.

`snap.shmim` lists streams to save as part of the calibration record, e.g. `aol1_wfsref,aol1_wfsmask`. Each must exist at startup.

Use these rather than `aol1_imWFS2`/`aol1_modevalWFS` themselves, to avoid stealing cache lines from the loop. Raw `camwfs` (the default) also works. The cacao WFS acquisition and modal extraction must be running with the loop open, and the preflight fails if a stream isn't updating. The modal coefficients are the same coordinates that `aol1_modevalWFS-sw` records, so a model calibrated on them can be applied directly to recorded science sequences.

The response for each camera is `(mean(+) - mean(-))/2`, averaged over repeats. It is **not** divided by the amplitude. The symmetric mean `(mean(+) + mean(-))/2` is saved too, so both sign means can be recovered. If the sign doesn't change between consecutive acquisitions (the middle `--`), the DM is not rewritten and no frames are discarded.

Every `ref.every` pokes (default 1), a zero-poke **reference** is acquired the same way, with the DM channel at its initial content. There is also one reference before the first poke. The **even part** `mean − ref` is formed with the reference linearly interpolated to each poke's center time. With `+--+` and a reference on each side, this cancels linear drift. In terms of `I = |C|² + |S|² + 2Re(C*S)`, the reference is |C|², the odd part (`resp`) is 2Re(C*S), and the even part is |S|² for the poked shape. `ref.every=0` disables references and the even part.

## Operator setup (on RTC)

- Loop open, system aligned, cameras running at their intended settings. Don't change camera modes or ROIs during a run.
- camsci1 must appear on RTC. On RTC run `shmimTCPreceive <port>`, then on ICC run `shmimTCPtransmit camsci1 <RTC address> <port>` (see the handbook's shmimTCP page).
- A poke cube: FITS, 50x50xNpoke, in microns, each shape normalized to 1 µm rms over the active actuator mask. xlinqCal does not check the normalization.

## Running

```
xlinqCal -c xlinqCal.conf            # see example/xlinqCal.conf
xlinqCal -c xlinqCal.conf --poke.amp=0.05 --poke.first=100 --yes
```

Before poking, xlinqCal checks the streams, measures both frame rates, and prints the estimated run time and output size. It then asks for confirmation (`--yes` skips this). Ctrl-C stops cleanly: the DM channel is restored, completed planes are written, and the log gives the `poke.first` to resume from. The DM channel is also restored on errors and timeouts. If you have a CPU set for non-RT readers, run the program inside it (`taskset`/`cset`).

With the defaults, each poke takes 58 camsci1 frames, about 14.5 s at 4 Hz: 3 DM changes × 2 discarded frames + 4 acquisitions × 10 averaged frames, plus a reference (2 discarded + 10 averaged). The preflight prints the total estimate.

## Outputs

In `out.dir`, with prefix `<out.prefix>_<UTC timestamp>` (e.g. `xlinqCal_20261121T073542`; `out.prefix` defaults to `xlinqCal`). The timestamp is set at startup, so reusing `out.prefix` starts a new data set instead of overwriting. `out.timestamp=false` drops it. A prefix already used by any file in `out.dir` is refused before the preflight, with or without the timestamp. The preflight summary shows the output prefix.

| file | content |
|---|---|
| `<prefix>_<cam>_resp.fits` | odd part `(mean(+) − mean(−))/2`, one plane per poke |
| `<prefix>_<cam>_mean.fits` | symmetric mean `(mean(+) + mean(−))/2`, one plane per poke |
| `<prefix>_<cam>_even.fits` | even part `mean − ref` (ref interpolated in time), one plane per poke. `EVEN1SID` counts planes that used only the nearest reference (e.g. after a stop) |
| `<prefix>_<cam>_ref.fits` | zero-poke reference frames, in acquisition order |
| `<prefix>_timing.fits` | one row per acquisition: poke, repeat, sign (0 = reference, with the reference index in REPEAT), DM cnt0 and projection, camsci1 cnt0 range and frame counts, gate times, the camsci1 acquisition times (stream `atime`) of the first and last averaged frames (`SCIATIMA`, `SCIATIMB`) and the maximum pixel value (`SCIMAX`). Then, for each WFS stream `j`: its cnt0 range, frame count, missed and torn frames, and first and last atimes (`WjCNT0A`, `WjCNT0B`, `WjN`, `WjMISS`, `WjTORN`, `WjATIMA`, `WjATIMB`). Times are Unix seconds; atimes are as written by the stream's producer (the camera frame timestamp for MagAO-X cameras). Column names are in `TCOLn` |
| `<prefix>_pokes.fits` | the applied pokes, `poke.amp` × the requested planes of `poke.file` [µm], with the poke file's header cards copied (copied numeric cards become strings) |
| `<prefix>_snap_<name>.fits` | content of each `snap.shmim` stream (e.g. `aol1_wfsref`, `aol1_wfsmask`) just before the first poke, saved as float; header has the stream's cnt0 and write time |
| `<prefix>_snap_<name>_end.fits` | written only if that stream changed during the run (a warning is logged) |
| `<prefix>_indi.txt`, `<prefix>_indi_end.txt` | with `indi.enable`: INDI state at the start and end, one `device.property.element=value` line per element (getINDI's format); unreadable properties are noted with `#` |
| `<prefix>_camsci1_dark.fits` | with `dark.n` > 0: mean of `dark.n` camsci1 frames taken with the shutter shut after a complete run |
| `<prefix>_camsci1_psf.fits`, `<prefix>_camsci1_psf_end.fits` | with `psf.n` > 0: PSF block before the first poke and after a complete run, a cube of `psf.n`-frame means (plane 0: `psf.filter`, DM at its initial state; with calibration speckles, also ± speckles at both filters; see `PLANEn`) |
| `<prefix>.log` | run log |

`<cam>` is the stream name (`camsci1`, and each WFS stream). Headers record the configuration, stream names, datatypes, measured fps, poke range (`POKE0`, `NPOKE`), references (`REFEVERY`, `NREF`), the camsci1 linearity limit and count (`SCILINMX`, `NNONLIN`), the INDI exposure time and EM gain (`SCIEXPT`, `SCIEMGN`) and dark frame count (`NDARK`), `PRODUCT`, `COMPLETE`, and start/end times. Partial results are checkpointed every `out.checkpointMin` minutes.

## INDI: system state and dark

xlinqCal is a minimal INDI client. It shells out to MagAO-X's `getINDI` and `setINDI`, and makes no INDI calls unless `indi.enable=true`.

- **State:** at the start and end it records the camera's `fsm`, `exptime`, `emgain`, `fps`, `shutter_status`, and `shutter` properties, plus `indi.props`, to `_indi.txt` / `_indi_end.txt`. A change in exposure time or EM gain during the run is logged as a warning.
- **Dark:** with `dark.n` > 0, the preflight requires `<sci.device>.shutter_status.status=READY` and the shutter open, and refuses to start otherwise. After a complete run, with the DM restored, it:
  1. shuts the shutter (`setINDI -s <dev>.shutter.toggle=On`) and waits for the readback;
  2. discards `dark.discard` frames, then averages `dark.n` frames;
  3. opens the shutter again and confirms it.

  Once shut, the shutter is always commanded open again, also after a stop or error; if that can't be confirmed, the log says so loudly. A stopped or failed run takes no dark.
- **MagAOX rules for INDI clients:** every get is one exact `device.property` (no wildcards, no device-wide or all-property requests). Every set is type-flagged. Each call has a wall-clock limit (`indi.timeout`), and readbacks are polled every `indi.poll` s (default 2, the MagAOX minimum).
- **Reachability on RTC:** `camsci1` is visible through RTC's indiserver. `fwsci1`, `stagesci1`, and `camsci1-dark` need to be added to RTC's indiserver remotes before they can go in `indi.props`.

## Saturated core with unsaturated PSFs

At the default light level the camsci1 halo is read-noise limited. Opening up the ND filter by 10–100× raises the speckle SNR by about as much, at the cost of a saturated PSF core. The core doesn't matter for the speckles or the even part, but the PSF itself is still needed (photometry, speckle shape), so xlinqCal can take it unsaturated with the ND wheel:

- **Core mask:** `sci.satRadius` [px] excludes a disk around the saturated core from the linearity check. The disk is centered on the preflight frame's pixels above `sci.linearMax`, then, after a PSF block with the wheel moved, on the unsaturated PSF (the centroid of the pixels above half its maximum), because the bleed stripe pulls the saturated pixels' center off the core (9 px in try6). `SCIMAX` (timing table) is then the maximum outside the mask, so blooming along columns or a brightening halo is still counted and warned. Headers: `SATRAD`, `SATCX`, `SATCY`, `SATCPSF` (1 if centered on the PSF).
- **PSF blocks:** with `psf.n` > 0 (needs `indi.enable`), the preflight reads `<psf.wheel>.filterName` (default wheel `fwscind`). It refuses to start unless exactly one filter is On, `psf.filter` (default `ND3_0`) exists, and the wheel is READY. Before the first poke, and again after a complete run, xlinqCal takes a PSF block and saves it as a cube of `psf.n`-frame means in `_camsci1_psf.fits` (`_psf_end.fits`):
  1. with calibration speckles and the wheel away from `psf.filter`: reference, +speckles, −speckles at the calibration filter (planes 3–5);
  2. if the wheel is away from `psf.filter`: sets `<psf.wheel>.filterName.<psf.filter>=On` (`setINDI -s`) and waits until that element reads On and the wheel is READY (it only turns On at the new position, so a premature READY is not accepted);
  3. with calibration speckles: −speckles, +speckles at `psf.filter` (planes 2, 1); then the reference, the PSF (plane 0);
  4. puts the wheel back to its original filter and confirms it. This always happens once the wheel was commanded, also after a stop or error; if it can't be confirmed, the log says so loudly. The DM is always restored.

  Each plane follows a DM change or wheel move, with the frame in progress and `psf.discard` more discarded. If the wheel is already at `psf.filter` (normal for a coronagraphic run), the block is taken in place, without moving it (planes 0–2). A failed start block stops the run before any poke. A stopped or failed run takes no end block. The header has `PLANEn` (filter and DM state), `PSFFILT`, `ORIGFILT`, `PSFMOVE`, `PSFBEG`/`PSFEND`, `PSFMAX` and `CALMAX` (both warned if above `sci.linearMax`), and the `SPK*` speckle settings.
- **Calibration speckles:** `psf.speckAmp` [µm] > 0 adds `speckAmp cos(2π n·x)` at `psf.speckSep` cycles across the DM and `psf.speckAngle` degrees from the DM first axis, plus the 90°-rotated copy if `psf.speckCross` (4 speckles), applied as + and − at each filter. (+ + −)/2 − reference is the speckle intensity without its interference with the static field or the dark, so the ratio of the speckle fluxes between the filters is the transmission. Choose the amplitude so the speckles are well above the noise at `psf.filter` but below `sci.linearMax` at the calibration filter, and the separation so they clear `sci.satRadius`.
- **Unsaturated low orders:** camsci1 bleeds along columns when saturated, which spoils the speckles nearest the core. With `psf.unsatLast` = N (needs `indi.enable`, and the wheel away from `psf.filter` at the start), pokes up to index N are measured with the wheel at `psf.filter`, and the rest at the calibration filter. Order the poke cube by spatial frequency so N splits low from high orders. The run moves the wheel to `psf.filter` after the start PSF block, and at the switch takes a reference on each side of the wheel move, so every poke's even part uses references at its own filter. The first acquisition after each move discards the frame in progress and `sci.nSettle` more. The wheel is put back if the run ends, stops or fails inside the segment. Headers: `UNSATLST`, `NREFUNS` (the leading references at `psf.filter`); the timing table has an `UNSAT` column. Use PSF blocks with calibration speckles to get the scale between the segments: `xlq_fourier.load_set` divides the unsaturated planes by T from `psf_flux.run_transmission`, and averages only the saturated-filter references.
- **EM gain guard:** with INDI on, xlinqCal refuses to run when a preflight frame is above `sci.linearMax` and `<sci.device>.emgain.current` > 1. Saturating with EM gain ages the EM register.
- **Flux ratio:** the transmission ratio between the two filters is not taken from the ND label. `tools/psf_flux.py <prefix>_camsci1_psf.fits --out psfcal.fits` measures it from the calibration speckles (round apertures, annulus background, speckles nonlinear at the calibration filter dropped), and writes the PSF in calibration-filter counts. `tools/dm_response.py --psf psfcal.fits --fit-center` then measures g(n) from a saturated run: it normalizes by that PSF, starts the axis fit at its centroid, and skips speckle apertures on saturated or bleeding reference pixels.

## Tools

`tools/shape_pokes.py` makes an amplitude-shaped poke cube from a previous data set. Each mode's amplitude is set so that its camsci1 speckle pair (the even part) carries about `--target` of the PSF flux, capped at `--cap` µm. It works from a smooth fit of the measured speckle energy against spatial frequency. It also fits the geometry from DM frequency to camsci1 pixels, and records the law and geometry in the header. The `AMP` and `FREQ` extensions hold the per-mode amplitude and frequency. Run xlinqCal with `poke.amp=1` on the output.

```
python3 tools/shape_pokes.py --modes modFourier_48.fits --data try1/xlinqCal_20261119_001 --out modFourier_48_shaped.fits
```

`tools/quadrature_field.py` uses each Fourier frequency's cos/sin mode pair as pairwise probes. The two odd parts give both quadratures of the coherent field C at the speckle, so it estimates C at every speckle peak with no DM gain needed. It prints the coherent fraction |C|²/ref by radius. It also inverse-transforms the C lattice into the pupil-plane field on the actuator grid, giving the pupil registered on the DM (amplitude) and the quasi-static phase seen by camsci1. It writes a FITS file with the field and a table of per-speckle results, plus an optional PNG.

`tools/test_2f.py` tests the second-order model. It predicts each low-order mode's even part at ±2f from the odd parts of the mode pair at 2f; the only free parameter is κ, in rad per µm of DM command. It also fits a DM 2f harmonic (from actuator-coupling nonlinearity), whose field is in quadrature with the optical term. κ and the harmonic γ are fit jointly over all windows, with noise correction and bootstrap errors. It reports both by side and |n|, plus κ from the |S₂|² ghost speckle. A negative κ means `--side-sign` (which speckle of a pair carries e^{+iψ}) is the twin. Both tools share `tools/xlq_fourier.py`.

```
python3 tools/test_2f.py --modes modFourier_48.fits --data try2/xlinqCal_20261119_002
python3 tools/quadrature_field.py --modes modFourier_48.fits --data try2/xlinqCal_20261119_002 \
    --out try2/field.fits --png try2/field.png
```

`tools/telem_screens.py` makes a poke cube from the AO residual of an observation, so the coronagraph's response to the observed wavefronts can be calibrated on the internal source afterwards (`example/runSky.conf`). It maps the full-rate telemetry to Fourier coefficients using a Fourier-mode run taken with the same WFS and reconstructor (`--cal`). The input is either the loop's modal coefficients (`--input modeval`, for any reconstructor including a nonlinear one) or raw camwfs frames (`--input camwfs`, linear reconstruction, modulated pyramid only). It reads the streamWriter `.xrif` archives directly (`tools/xrif_read.py`, libxrif through ctypes), searching directories recursively and selecting the observation by atime with `--start`/`--end`, so no decoding to FITS is needed. The lag-1 covariance, over frames with consecutive cnt0, gives the signal covariance with the WFS noise removed. The cube holds the leading PCA modes of that covariance at 1σ, then random screens drawn from it at several scales, then repeats. Every plane's Fourier coefficients are in the `COEFFS` extension.

```
python3 tools/telem_screens.py --input modeval /data/rawimages/aol1_modevalWFS \
    --start 2026-11-25T03:10:00 --end 2026-11-25T03:40:00 \
    --modes modFourier_48.fits --cal try10/try10_<time> --out sky_screens.fits
```

## Building and testing

Requires mxlib (built with MILK support), ImageStreamIO, and cfitsio, found via `pkg-config mxlib`.

```
make            # xlinqCal and xlinqSim
make test       # unit tests + simulator integration tests
```

`xlinqSim` creates `xlsim_dm01disp04`, `xlsim_dm01disp`, `xlsim_camwfs`, and `xlsim_camsci1`, with known linear responses. It also writes `xlsim_wfsR.fits`, `xlsim_sciR.fits`, and `xlsim_pokes.fits`, so xlinqCal can be tried on a workstation:

```
./xlinqSim &
./xlinqCal --dm.channel=xlsim_dm01disp04 --dm.comb=xlsim_dm01disp --sci.shmim=xlsim_camsci1 \
           --wfs.shmim=xlsim_camwfs --poke.file=xlsim_pokes.fits --poke.amp=0.1 --out.dir=simout
```
