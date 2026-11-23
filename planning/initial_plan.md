
I want to build a standalone application that works with the MagAO-X instrument to take calibration data.
For reference /home/jrmales/Source/MagAOX (dev branch) serves as the specification for the interface, and we need to follow operations rules in its AGENTS.md.  

The intended calibration process will be:
- with loop open (but operator will have system aligned and cameras ready)
- Poke dmtweeter by writing a shape to MILK shmim dm01disp04
- listen on camsci1 MILK shmim semaphore to synchronize
- record camwfs and camsci1 images such that a syncrhonized sequence of camsci1 frames and camwfs frames are averaged.  
- note that camwfs will be at ~2kHz and camsci1 will be at ~4 Hz.
- then invert the poke and take same sequence
- make the poke response (plus-poke - minus-poke)/2 for each camera
- repeat through sequence of pokes
- save FITS cubes with the sequence of poke responses

At this point it's just managing the MILK interface and taking data, I don't forsee a need to use INDI.  We'll rely on the operator to set it up.

Please analyze and make a plan below.  Ask for clarification if needed.

---

# Analysis and Plan

*(Added 2026-11-19 by Claude; based on MagAOX `dev` @ ccc26f4f and `/opt/MagAOX/config`. Sections 1-4 are the original analysis; where they conflict with section 5 (decisions after Q&A), section 5 wins.)*

## 1. Key findings from the MagAOX spec

### 1.1 The streams are on two machines
From the proclists in `/opt/MagAOX/config`:

| stream | producer | host |
|---|---|---|
| `camwfs` (OCAM2K, `bin2` default, up to 3.6 kHz) | `ocam2KCtrl` | RTC |
| `dm01disp04` (one channel of `dmtweeter`, BMC 2k, 50x50 float) | `bmcCtrl` | RTC |
| `camsci1` (EMCCD) | `picamCtrl` | ICC |

MILK shmims are local to one host, so no single process can open all three directly. Options:

- **A (recommended): run on RTC.** Mirror `camsci1` to RTC with the existing milkzmq client `mzmqClientRTC_ICC` (server `mzmqServerICC_RTC`, port 5558, already configured for `camsci1,camsci2,camtip`). Neither process is in the normal proclist, so the operator would start them as part of setup. The milkzmq client keeps `cnt0` and `writetime` (`milkzmqClient.hpp:525-527`) but **not** `atime`, and it sets `cnt1=0`. At 4 Hz, milkzmq adds a few ms of latency, which is about 1% of a 250 ms camsci1 frame.
- B: run on ICC and mirror `camwfs` and `dm01disp04` there. Rejected: mirroring a 2 kHz camera through milkzmq will drop frames, and we would be writing the DM over the network.

### 1.2 Existing code that already solves most of this
`libMagAOX/app/dev/dmPokeWFS.hpp` (the base class behind `dmPokeCenter` and `dmPokeXCorr`) already handles poking a DM channel and syncing it to camera frames:
- It opens the DM channel with `mx::improc::milkImage<float>` in passive mode. Assigning an Eigen image to it writes the image and posts the semaphore.
- It waits on the combined `dm01disp` stream semaphore (`waitForDmTrigger`) to confirm the DM summed the new command. It claims a free semaphore with `ImageStreamIO_getsemwaitindex` and releases it on close (`semReadPID[...] = 0`).
- `basicTimedPoke()` does the sequence: write ±poke, wait for the trigger, sleep `dmSleep`, flush, skip `nSettleImages`, then accumulate `nPokeImages` with sign. It repeats this `nPokeAverage` times, so `(plus - minus)` builds up in a single sum.
- It writes FITS with `mx::fits::fitsFile`.

That code is a `MagAOXApp` (it needs INDI and the logger) and handles only one camera, so we can't reuse it directly. We should copy its approach and its ImageStreamIO semaphore handling. `bmcCtrl` also publishes `completionStream=dmtweeter_done`, which we could use as a stronger "DM applied" signal than the combined-channel semaphore.

### 1.3 Dark/bias cancel out
The response is `(plus - minus)/2`, so the dark and bias drop out of it. No dark is needed for the response cubes. The `camsci1_dark` stream is not mirrored to RTC anyway. We only need a dark if we also save mean (reference) images (see open questions).

### 1.4 AGENTS.md constraints that apply
- Rule 0: agents never control hardware. This app runs on RTC (or anywhere the operator chooses), started by a person. I develop and test only against simulated shmims on a workstation (`mx11` is fine).
- Rule 28: `camwfs` is a loop stream. The loop is open during calibration, but this still adds a reader. Use semaphore waits (no spinning), copy each frame out once, and accumulate with vectorized operations. Make the cpuset configurable.
- Style rules 1-13, 21, 26 (Doxygen, `m_` members, Allman braces, header-only class with a `.cpp` holding only `main`), plus clang-format.

## 2. Proposed design

### 2.1 Form
A standalone C++17 command-line program, `xlinqCal`, built on:
- **mxlib**: `mx::app::application` for config file and command line (same format as MagAOX `.conf` files), `milkImage`, `eigenImage`/`eigenCube`, and `fitsFile`/`fitsHeader`.
- **ImageStreamIO** directly, for semaphore index claiming and release, `semflush`, and reading frame metadata (`cnt0`, `writetime`, `atime`). The pattern is the same as in `dmPokeWFS`.
- No INDI and no MagAOX logger. Messages go to stdout/stderr, plus a run log written next to the output.

Layout: `xlinqCal.hpp` (class declaration plus out-of-class inline definitions), `xlinqCal.cpp` (`main` only), `Makefile`, `tests/` (Catch2, same structure as MagAOX app tests), and `sim/` (stream simulator, see 2.6).

### 2.2 Inputs (config file / CLI)
- `dm.channel` (default `dm01disp04`), `dm.triggerChannel` (default `dm01disp`, or `dmtweeter_done`), `dm.settleTime`
- `poke.file`: a FITS cube of DM shapes, 50x50xNpoke. This can be any basis: single actuators, Hadamard, Fourier, or modes. `poke.amp` is a scale factor. `poke.first`/`poke.last` select a subrange so an interrupted run can be resumed.
- `sci.shmim` (default `camsci1`), `sci.nSettle` (frames discarded after each poke, default 1 or 2), `sci.nFrames` (frames averaged per poke sign)
- `wfs.shmim` (default `camwfs`), `wfs.cpuset`
- `seq.nRepeat` (number of ± cycles averaged per poke), `seq.order` (`+-` or `+--+`; `+--+` cancels linear drift)
- `out.dir`, `out.prefix`, `out.checkpointEvery`

### 2.3 Threads and synchronization
- **WFS thread**: waits on its own `camwfs` semaphore and copies each new frame out once (converting to float from the native type). If the gate is open, it adds the frame to the current accumulator and increments a count. It also records the first and last `cnt0` and the timestamps for each window.
- **Main (sequencer) thread**, for each poke `k` and each sign `s` in the sequence order:
  1. Write `s*amp*shape_k` to `dm01disp04` (copy, then post).
  2. Wait on the `dm01disp` (or `dmtweeter_done`) semaphore, then sleep `dm.settleTime`.
  3. Flush the `camsci1` semaphore, then skip `sci.nSettle` camsci1 frames. The frame being exposed during the poke is contaminated, and the mirror adds a small delay.
  4. On the next camsci1 post, **open the camwfs gate**. Then add each of the next `sci.nFrames` camsci1 frames to the camsci1 accumulator. On the post of the last of these frames, **close the gate**.

     Each camsci1 semaphore marks the end of a frame. With a frame-transfer EMCCD in continuous mode, exposure ≈ frame period. So the camwfs frames between those posts cover the same interval as the averaged camsci1 exposures, offset by the readout plus mirror latency (a few ms).
  5. Divide each accumulator by its frame count to get the mean for this sign, and add `s * mean` to that camera's response sum.
- After all signs and repeats: `response = sum / (2*nRepeat)` (or divide by the actual number of ± pairs), stored in plane `k` of each output cube.
- After each poke and on any exit path (error, timeout, SIGINT/SIGTERM), restore `dm01disp04` to the contents saved at startup. Default: warn if it was nonzero at startup.

**Later upgrade:** select camwfs frames by timestamp instead of by gate. Keep a ring buffer of camwfs frames with `writetime`, and match them to camsci1 `writetime − texp`. This removes the latency offset, but it needs ICC and RTC clocks synced to well under a millisecond. Either way, the per-window timestamps and `cnt0` ranges are recorded, so we can check alignment afterwards.

### 2.4 Outputs
- `<prefix>_camsci1_resp.fits`: cube, nx×ny×Npoke
- `<prefix>_camwfs_resp.fits`: cube, nx×ny×Npoke
- Optional (see questions): mean "reference" cubes or a single mean image per camera
- `<prefix>_pokes.fits`: copy of the applied shapes times amp (or a header pointer to `poke.file`)
- `<prefix>_timing.fits` (binary table or simple image): per poke and per sign, the camsci1 `cnt0` range and number of frames, the camwfs `cnt0` range and number of frames, and the window start and end times. This makes dropped frames or misalignment visible.
- FITS headers: full config, stream names and sizes, measured fps of both cameras, amp, nSettle/nFrames/nRepeat, start/end UTC, git hash.
- Checkpoint the partial cubes every `out.checkpointEvery` pokes, so a crash or abort on a 30+ minute run doesn't lose everything.

### 2.5 Pre-flight checks (before poking)
- All streams open, with the expected datatypes and sizes. The DM channel must be 50x50 float, matching `poke.file`.
- Both cameras are updating. Measure fps over about 2 s; fail with a clear message if camsci1 is stale, e.g. the mirror isn't running. Report the expected number of camwfs frames per camsci1 frame.
- Print an estimated run time and wait for operator confirmation (`--yes` skips this).

Time budget example: Npoke=2040 single actuators, nSettle=1, nFrames=1, nRepeat=1, `+-`: 2040 × 2 × 2 × 0.25 s ≈ **34 min**. A smaller Hadamard or modal basis, or more frames per poke, changes this linearly.

### 2.6 Testing without hardware
- `sim/xlinqSim`: creates `camwfs` (about 2 kHz), `camsci1` (about 4 Hz), and `dm01disp`/`dm01disp04` shmims locally, using test names. Each camera frame is a known linear response to the DM channel plus noise, offset, and a slow drift. Camsci1 frames integrate over their exposure window, so we can check the gating and the drift cancellation.
- Catch2 unit tests: poke-sequence generation, ± accumulation math, type conversion, gating counts, DM restore on error and abort.
- Integration test: run xlinqCal against the simulator and compare the recovered response cubes with the injected response matrix.
- Final validation on the instrument is done by the operator (not the agent), with a short poke list first.

## 3. Implementation steps
1. Repo skeleton: Makefile against mxlib + ImageStreamIO + cfitsio, `.clang-format` copied from MagAOX, git init with branch naming per AGENTS.md (if this becomes a repo).
2. Simulator (`sim/`), so everything after this can be tested locally.
3. Stream wrappers: open, claim and release semaphore index, timed semaphore wait, copy-to-float for any ImageStreamIO type. Check for reusable helpers in MagAOX `ImageStreamIO/pixaccess.hpp` and `semUtilsDerived.hpp` first, or propose moving them to mxlib.
4. WFS gated-accumulator thread.
5. Sequencer: DM write and restore, trigger wait, camsci1 settle and accumulate, signal handling.
6. Output and checkpointing, plus timing table and headers.
7. Pre-flight checks and run-time estimate.
8. Tests (unit + sim integration), Doxygen, clang-format; a plan update recording decisions.
9. Operator test on RTC with a few pokes; tune `nSettle` and `settleTime` from the timing table.

## 4. Open questions
1. **Host and mirroring:** OK to run on RTC with `camsci1` coming through the `mzmqClientRTC_ICC` milkzmq mirror? Or do you have another transport in mind?

Yes to running on RTC. We'll use shmimTCP, which has the same effect.  camsci1 frames will show up on RTC with relatively low latency and no frame drops.

2. **Poke definition:** is a FITS cube of 50x50 shapes times a scalar amplitude the right input? What units are used on `dm01disp04` (microns of surface)? Should the app generate standard sets itself (single actuators within the `bmc_2k` mask, Hadamard)?

Yes to input. Units are microns. Let's assume they are normalized to rms=1 micron and I'll supply an amplitude

3. **Normalization:** keep `(plus - minus)/2` exactly as you wrote it, or divide by amplitude as well, to get response per unit command?

I should have stated the goal up front: we want to be able to use the WFS images to predict the response in the science images.  So the eventual use case will be to multiply a given wfs image by the response matrix to generate a predicted science frame (there's more to it than just a simple projection but this is the idea).  So I think that we don't want to divide by amplitude because it will be included in the response.

4. **Defaults:** typical `nFrames`, `nSettle`, and `nRepeat` for camsci1? Should the `+--+` drift-cancelling order be the default?

nSettle should be one frame after the first frame following the poke.
nFrames should be 10
nRepeat should be 2

5. **Extra products:** do you want mean (unpoked or `(plus+minus)/2`) images saved per camera? If so, should darks come from files, or from a dark taken by the app with the operator closing the shutters?

No extra produces.

6. **Window semantics:** is averaging camwfs over exactly the camsci1-frame windows (gate method, ~ms latency error) good enough to start with? Are ICC and RTC clocks synced well enough for the timestamp method later?

The only thing that really matters is that the poke is on the dm for the duration of the averaging.  So ms, even a few ms, is just handfuls of camwfs frames relative to O(100) per camsci exposure.

7. **Language and home:** C++/mxlib (my recommendation, given the 2 kHz reader) or Python (magpyx/pyMilk)? Is `xlinqCal` its own repo, or should it eventually become a MagAOX app or util (which would let it reuse `dmPokeWFS` pieces directly)?

C++/mxlib, standalone from MagAOX for now.

8. **Camera config drift:** should the app refuse to run, or only record it, if camwfs mode/ROI or camsci1 ROI differs from an expected value given in the config?

Just record it.

---

## 5. Decisions (2026-11-19, after Q&A)

### 5.1 Goal
The response cubes will be used to predict camsci1 images from camwfs images (roughly `sci ≈ M · wfs`, built from the two cubes). Because of that, responses are **not** divided by amplitude. Amplitude is part of the measured response and is recorded in the headers.

### 5.2 Answers applied
| # | Decision |
|---|---|
| 1 | Runs on **RTC**. camsci1 reaches RTC through **shmimTCP** (`shmimTCPtransmit camsci1 <RTC> <port>` on ICC, `shmimTCPreceive` on RTC), with low latency and no frame drops. The operator sets this up. The app just opens a local stream (name configurable, default `camsci1`). |
| 2 | Input is a FITS cube of 50x50xNpoke shapes in **microns**, each normalized to **1 µm rms**. `poke.amp` (µm rms) is supplied at run time. The app does not generate bases. |
| 3 | `resp = (mean(+) − mean(−))/2`, averaged over repeats. No division by amplitude. |
| 4 | After each DM write, the **first camsci1 frame that follows the poke is always discarded**, then `sci.nSettle` more frames (default **1**). Then `sci.nFrames` (default **10**) frames are averaged. `seq.nRepeat` defaults to **2**. Sign order alternates per repeat (`+-`, then `-+`, i.e. `+--+` for 2 repeats), which cancels linear drift. |
| 5 | No extra products: no reference/mean images and no darks. |
| 6 | Gate method only. What matters is that the poke is on the DM for the whole averaging window, and a few ms is a handful of camwfs frames out of O(500) per camsci1 frame. The timestamp method is dropped. |
| 7 | C++/mxlib. **Its own git repo** (`xlinqCal`), standalone from MagAOX. |
| 8 | Camera configuration is only recorded: stream sizes, datatypes, and measured fps go into the headers. |

### 5.3 Lessons from MagAOX `agents/plans/2026-09/dm-poke-sign-synchronization.md`
- `dmcomb` republishes `dm01disp` only when the sum of channel `cnt0` advances. `dmPokeWFS` writes its channel **passively**, so it relies on the closed loop to trigger `dmcomb`. Here the loop is open, so xlinqCal must write `dm01disp04` **non-passively** (`ImageStreamIO_UpdateIm`, which increments `cnt0` and posts semaphores).
- A `dm01disp` semaphore wake does not prove that our command was included. After the wake, xlinqCal computes the projection of `(dm01disp − dm01disp_before)` onto the commanded change and records it per poke. It warns if the projection is far from 1. This is a diagnostic only, never a gate.
- The `dmPokeXCorr` WFS monitor missed about 16% of frames and could reuse a buffer. xlinqCal copies each camwfs frame once per wake and checks `cnt0` before and after the copy. Torn frames (`cnt0` changed during the copy) are dropped and counted, gaps in `cnt0` are counted as missed, and only frames actually accumulated go into the mean. All counts are saved.
- Settling is based on frames, not sleeps. Each camsci1 frame is 250 ms, which is much longer than the ms-level DM/dmcomb latency, so the discarded first frame plus `nSettle` covers it.

### 5.4 Run time with the defaults
Per poke: 2 repeats × 2 signs × (1 + 1 + 10) frames / 4 Hz = **12 s**. For 2040 single actuators that is ≈ 6.8 h; for 500 modes, ≈ 1.7 h. Lowering `nFrames` or `nRepeat` scales this linearly. The preflight prints this estimate and the output size, and asks for confirmation.

### 5.5 Repo layout
```
xlinqCal/
  Makefile               builds xlinqCal and xlinqSim; `make test` builds and runs tests
  .clang-format          copied from MagAOX
  AGENTS.md              points to MagAOX AGENTS.md rules, plus local notes
  README.md              operator usage
  example/xlinqCal.conf
  src/isioStream.hpp     ImageStreamIO reader: open, semaphore claim/release, timed wait, typed copy to float/double
  src/xlinqCal.hpp       application class (header-only, out-of-class inline definitions)
  src/xlinqCal.cpp       main only
  sim/xlinqSim.hpp       stream simulator: dmcomb + camwfs + camsci1 with known linear responses
  sim/xlinqSim.cpp       main only
  tests/                 Catch2 unit tests + in-process simulator integration test
  planning/
```

### 5.6 Safety
- The simulator refuses to create stream names that don't start with a test prefix (default `xlsim_`), so it can never clobber real instrument streams if run on RTC.
- At startup, xlinqCal saves the content of `dm01disp04` and restores it on every exit path (normal end, error, timeout, SIGINT/SIGTERM). It warns if the channel was nonzero at startup.
- A camsci1 or dm01disp semaphore timeout aborts the run, restores the DM, and writes the completed planes. `poke.first`/`poke.last` let a run be resumed.

### 5.7 Execution notes
- 2026-11-19: first implementation complete; all tests pass (`make test`: 7 cases, 111 assertions).
  - `src/isioStream.hpp`: ImageStreamIO reader. Claims a semaphore with `ImageStreamIO_getsemwaitindex` and releases it on close. Timed waits use `sem_timedwait`. Copies to float/double for all integer and float types. Torn frames are detected by `cnt0` changing during the copy.
  - `src/xlinqCal.hpp`: `mx::app::application` subclass. The DM channel is `mx::improc::milkImage<float>`, non-passive. The command is `initial + sign*amp*shape`, so a nonzero channel is preserved and then restored. The WFS thread accumulates under a mutex-protected gate. Sci frames are accepted only when `cnt0` advances.
  - Change from 5.2 #4: when the sign doesn't change between consecutive acquisitions (the middle `--` of `+--+`), the DM is not rewritten and no frames are discarded. The gate opens at the post that closed the previous window. This saves 2 of 48 frames per poke, and the `+--+` window centers stay symmetric, so linear drift still cancels. Per-poke time with the defaults is 46 frames ≈ 11.5 s (5.4 said 12 s).
  - The timing table is a 16 x Nacq double FITS image with `TCOLn` column names. A FITS binary table wasn't used because mxlib's `fitsFile` writes images only.
  - `sim/xlinqSim.hpp`: dmcomb, camwfs (uint16), and camsci1 (float) with random linear responses. camsci1 integrates the DM state over each exposure and posts `sciPostDelay` after the exposure ends. Without that delay, xlinqCal's pokes (made right after a camsci1 post) land at an exposure boundary, and the test could not detect a missing settle discard. A mutation check confirmed this: with the discard removed, the test passes without the delay, and with the delay it fails with 14.5% sci error.
  - Integration tests run xlinqCal in-process against the simulator. They check recovered responses against `amp*R*shape` (<5%), timing-table consistency, DM restore, FITS output, and a mid-run stop (exit code 1, completed planes written, nonzero initial channel restored, `POKE0`/`NPOKE`/`COMPLETE` headers).
- Not implemented from the original plan: `wfs.cpuset` (run the program inside a cpuset instead, see README); the timestamp-matching mode (dropped per 5.2 #6).

- 2026-11-19: removed the 1 µm rms check on poke shapes. The rms is defined over the DM's active actuator mask, so a full-frame rms check would warn on correctly normalized cubes. Supplying a correctly normalized cube is the operator's responsibility.

- 2026-11-19: added sign means and interleaved zero-poke references for the even part (|S|²; see `planning/linq_analysis.md` §9). Outputs per camera are now `resp` (odd), `mean` (symmetric), `even` (mean − ref, with ref linearly interpolated to the poke's center time), and `ref`. `ref.every` (default 1) sets the reference cadence, and there is always one reference before the first poke. The first acquisition always settles, so the gate is aligned to a camsci1 frame boundary. The simulator gained an optional camsci1 term `|H c|²` (`sciQuadGain`). The integration test checks the camsci1 even part against `amp²|H s|²` (<5%) and the camwfs even part against zero (<2% of the odd part) with drift present. A mutation check showed the test catches missing time interpolation (camwfs even/odd 0.67). Default per-poke time rises to 58 frames ≈ 14.5 s.

- 2026-11-19: multiple WFS streams. `wfs.shmim` is now a list; each stream is a `wfsChannel` (`src/wfsChannel.hpp`) with its own reader thread, gate, products, and timing columns (`W<j>…`). The intended configuration is `aol1_imWFS2pub,aol1_modevalWFSpub` (see `linq_analysis.md` §9.9). The simulator gained an optional modal-coefficient stream (`modevalName`/`nModes`, `sim.nModes`). The integration test gates camwfs and the modal stream together and checks both odd responses (<5%) and even parts (≈0). Per the user, `aol1_wfsrefc` is never updated and `wfsref` changes rarely on MagAO-X, so no reference-freeze check was added.

- 2026-11-19: snapshot streams. `snap.shmim` (example config: `aol1_wfsref,aol1_wfsmask`) lists streams opened at startup (missing = error), written to `<prefix>_snap_<name>.fits` before the first poke, and re-read after the run. A changed stream is also saved as `_snap_<name>_end.fits` with a warning. The code default is empty, so xlinqCal still runs without cacao. Tests cover an unchanged pair (contents verified, no `_end`) and a stream changed mid-run (`_end` written).

- 2026-11-19, first instrument data set (`try1/`, camsci1 at 26.5 Hz, `nFrames=2`, `nRepeat=1`, 2400 Fourier modes at 0.05 µm). Bookkeeping was clean: DM projection 1, no missed sci frames, WFS missed/torn ≈ 0, reference flux drift 0.13%. Even-part SNR ranged from about 450 (low order) to about 10 (corners). The modes are integer cycles per **50** actuators. The camsci1 geometry is about 4.43/4.27 px per cycle/50 act, rotated −27.6°, with 0.7 px fit residual. The amplitude was too large at low order (speckle pair ≈ 20% of PSF flux; WFS even/odd ≈ 0.5). Changes:
  - The doubled-stream-name check: mxlib joins repeated config keys, which caused the first failed start.
  - `<prefix>_pokes.fits`: applied pokes with the poke file's header.
  - `tools/shape_pokes.py`: per-mode amplitude law targeting about 1% speckle energy, capped at 0.05 µm. It gives 0.010 µm at low order to 0.05 µm in the corners, with projected even SNR about 17–24 for |n| < 24 at try1 settings.
  - `try*/` is ignored by git.

- 2026-11-19, second data set (`try2/`, shaped cube `modFourier_48_shaped.fits` with `poke.amp=1`, `nFrames=1`, bump mask in the upstream pupil). It took 13.5 min. Bookkeeping was clean: applied pokes equal the shaped cube, DM projection within 2×10⁻⁶ of 1, no missed sci frames, no torn WFS frames. WFS frames per window ranged 50–99 (median 75): camsci1 posts arrive with periodic jitter. `modevalWFSpub` stalled once (17 frames in one window). The apparent 3–7% ref flux jitter is a per-frame camsci1 bias offset (~1.3 counts); flux within 15 px of the core is stable to 0.09%. Results:
  - **Shaping worked:** speckle energy is 1.26–1.33% per mode for |n| < 24 (±4%), even peak SNR ≈ 23, energy SNR ≈ 75, minimum 9. One frame per sign suffices for single-mode |S|² at this target. Energy is 1.3× the target at all |n|, consistent with the sharper PSF behind the bump mask. The plate scale changed by +2.2% (cause unknown), so the geometry is refit per data set.
  - **WFS:** the low-order odd response per µm is 15–20% higher than in try1, so 0.05 µm compressed the WFS response. Odd SNR per mask pixel is ≈ 11, and the WFS even part is at the noise floor. Half of the WFS reference noise is slow and spatially coherent (seconds, top 5 PCs); the white half rose 1.4× per frame (the bump mask may cost camwfs light). Operator plan: more camwfs EM gain, with a new dark.
  - **Time:** each acquisition uses 1 of 3 camsci1 frames (poke frame discarded, plus `nSettle=1`). Proposed test: 200 modes with `nSettle=0` vs 1. If they agree, `nFrames=2` fits in the same time.
  - **New tools:** `tools/quadrature_field.py` and `tools/test_2f.py`. Results are in `linq_analysis.md` §9.12.

- 2026-11-20: acquisition times and a camsci1 linearity check.
  - **Timing table:** gains the stream `atime` of the first and last averaged frame for camsci1 (`SCIATIMA`/`SCIATIMB`) and each WFS stream (`WjATIMA`/`WjATIMB`). Clocks across MagAO-X computers are synced (user), so calibration and science data can be matched by atime and exposure time.
  - **atime source:** `isioStream::copy` reads atime inside the torn-frame check. It uses the per-slice atime array only for temporal circular buffers, as MagAOX `shmimMonitor` does.
  - **Simulator:** its cameras post with an explicit CLOCK_REALTIME atime (camsci1: end of exposure; modeval: the camwfs frame's atime).
  - **Linearity:** camsci1 is linear to 37k counts (user). `sci.linearMax` (default 37000, 0 = off) flags acquisitions with any averaged frame above it. Each acquisition's maximum goes in `SCIMAX`; the count goes in `NNONLIN` and the log. The preflight reports the maximum pixel of one frame. In try1, about half of the references had peaks of 37.1–37.9k; try2 stayed below 37.0k.
  - **Geometry (user):** the K-mirror keeps the pupil fixed on the DM, so the pupil-mask and DM-grid orientations relative to camsci1 are fixed. The measured rotation is −27.5°, stable to 0.1° between runs.
  - **INDI client (user):** make xlinqCal an INDI client by shelling out to `getINDI`/`setINDI`, to record system state and close the shutter for a dark. Details are in the next entry.
  - **Online modal basis:** `CMmodesDM.fits` (the online aol1 modes; 1563 modes, the 2024-11-13 `fklcombo_z10_2400…` calibration) was copied into the repo root (untracked). See `linq_analysis.md` §9.13.

- 2026-11-20: INDI state recording and a dark (`src/indiClient.hpp`). Interface facts are from a read-only survey of MagAOX `dev` (`ed8b5697`) and `/opt/MagAOX/config` (`onsky`):
  - **getINDI cost:** `getINDI dev.prop` sends a named getProperties and exits when the property arrives. Wildcards or `dev` alone make every app re-send all its defs and subscriptions (MagAOX incident 2026-03-11; plan `indiserver-chained-loop`). xlinqCal therefore only gets exact `device.property` names (validated: `[A-Za-z0-9_-]` components) and never dumps all properties.
  - **setINDI:** an untyped `setINDI` does a device-wide getProperties and silently refuses to send unless every element is listed. With `-s`/`-n` it sends at once and exits 0 when sent, not when applied. `-w` hangs on properties that settle Idle and exits 0 on Alert. xlinqCal uses `-s` and confirms by readback, polling no faster than `indi.poll` (default 2 s; MagAOX asks for ≥ 2 s between named getProperties).
  - **Calls:** run without a shell (fork/exec), with stdout/stderr captured and a wall-clock kill (`indi.timeout`, default 8 s).
  - **Shutter (MagAOX stdCamera + dssShutter):**
    - `camsci1.shutter.toggle` On = shut, Off = open. The command is explicit, not a flip, and asynchronous: a pulse plus sensor poll, up to 2 × 2 s.
    - Failures are only logged, and the property doesn't go to Alert.
    - The readback Off also means unknown or power-off, and one power-on code path reported the polarity reversed (since fixed in MagAOX). So xlinqCal trusts the readback only with `shutter_status.status=READY`, and checks both in the preflight. The dark sanity check warns if the dark maximum is above half the preflight maximum.
    - camsci1 runs `picamCtrl` on ICC. It is visible on RTC via `camsci1@rtc_icc_indi`.
  - **Dark placement:** after a complete run, with the DM restored, so a shutter problem can't spoil the calibration, and the dark uses the run's exposure settings. The shutter is always commanded open again once shut. A stopped or failed run takes no dark.
  - **Simulator and tests:** `shutterFile` (content `On` = shut) gives `sciDark` frames. The tests use fake `getINDI`/`setINDI` scripts over a state directory, applying sets after 0.3 s like an asynchronous device. They check:
    - the state files and header (`SCIEXPT`, `NDARK`);
    - the dark level;
    - the shutter left open;
    - exactly two typed shutter sets;
    - that every get is one exact `device.property`;
    - refusal with the shutter shut, not READY, or with `dark.n` but no `indi.enable`.
  - **Resolved:** the shutter polarity reversal at camera power-on (MagAOX `stdCamera.hpp`, commit `ec9247bc`) was fixed in MagAOX by the operator; the READY check stays as a guard. `fwsci1`, `stagesci1` and `camsci1-dark` are wanted in the run record and will be added to RTC's indiserver remotes by the operator, then listed in `indi.props`. First instrument test: a short run with `dark.n=5` and small `poke.last`, watching the shutter.
- **No overwrites (2026-11-21):** an operator test overwrote an earlier data set by reusing `out.prefix`. The prefix now always gets `_<UTC timestamp>` (set at startup, before the preflight), and `out.timestamp=false` turns that off. Independently, xlinqCal refuses to start if any file in `out.dir` is named `<prefix>_*` or `<prefix>.*`. It checks before opening streams, and again before writing, in case files appeared during the confirmation. A refused run writes nothing and never touches the DM. Resuming with `poke.first` therefore makes a new data set; the analysis tools take one prefix per set. Tested: a repeat of the main integration run with the same fixed prefix is refused, and the files are unchanged; the stop test checks the timestamped name.
- **Saturated core with unsaturated PSFs (2026-11-21):** the halo is read-noise limited (try3: 3.5 counts/px/frame at 45–65 px, against > 10 counts of noise), so 10–100× more light (less ND) gives about that much more speckle SNR, with the core saturated. The user chose the ND wheel only, with PSFs at the start and end.
  - **Why not exposure time:** camsci1 at 36 ms is at its readout-limited minimum (picamCtrl raises shorter requests to `ReadOutTimeCalculation`), so a shorter PSF exposure is not available. A longer saturated exposure adds no photons per second. Exposure changes are also online in picamCtrl, with `exptime.current` updating before new-exposure frames arrive, and with `synchro` on they restart camsci2 (MagAOX survey).
  - **Wheel:** fwscind runs `hsfwCtrl` (stdMotionStage). `filterName` is a one-of-many switch whose On element follows the actual position. `fsm.state` is READY when not moving; a 1 Hz poll can report READY before motion starts, so completion requires the target element On *and* READY. Move time is not documented.
  - **Implemented:**
    - config `psf.n/wheel/filter/discard/wait` and `sci.satRadius`;
    - indiPreflight now runs before the stream preflight, so the wheel state and EM gain are known for the summary and the guard;
    - preflight requires one filter On, different from `psf.filter`, and the wheel READY;
    - `takePsf` at the start (before any poke; failure aborts) and after a complete run (before the dark), always restoring the wheel;
    - core mask centered on the saturated pixels, with `SCIMAX` outside it;
    - EM gain guard (saturated with EM gain > 1 refuses).
  - **Tests:** the simulator has a Gaussian core, a saturation level, and an attenuation file following the fake wheel; the fake setINDI handles one-of-many switches. Tested: PSFs unsaturated at the attenuated level, refs saturated, `nNonlin` 0 with the mask and > 0 without, exactly four typed wheel sets, wheel restored, refusals (wheel already at the PSF filter, EM gain 10, no `indi.enable`).
  - **Open:** blooming along columns at 10–100× saturation is unknown for camsci1 (handbook: wells can saturate from 40k counts; ADC at 65535). Check the first saturated data set: SCIMAX outside the mask, and the columns through the core in the even part.
  - **Revised (user, 2026-11-21):** fwscind is fast, so `psf.wait` defaults to 10 s. A wheel already at `psf.filter` no longer refuses: that is normal for a coronagraphic run, and the block is taken in place. Flux calibration uses DM speckles present at both filters, as usual on MagAO-X: `psf.speckAmp/speckSep/speckAngle/speckCross` (a dmSpeckle-like sinusoid, crossed for 4 speckles) applied as + and − at each filter, so (+ + −)/2 − reference cancels the interference with the static field and the dark (the same even-part logic as the pokes; dmSpeckle modulates for the same reason). The block is a cube: PSF-filter reference/+/− in planes 0–2, then calibration-filter reference/+/− in 3–5 when the wheel moves; order cal ref, +, −, wheel, −, +, ref, so the DM changes once across the move and ends at its reference. Each plane discards the frame in progress plus `psf.discard`. `tools/psf_flux.py` gives T = speckle flux ratio and the PSF in calibration-filter counts (checked on a synthetic block, T = 0.0108 ± 0.0007 for 0.01). Tests: 6 planes with the sim's quadratic term as the speckle signal, ratio = attenuation within 3%; in place 3 planes, no wheel sets; 1 plane without speckles; `speckAmp` without `psf.n` refuses.
  - **Unsaturated low orders (user, 2026-11-21: camsci1 bleeds when saturated):** `psf.unsatLast` = N measures pokes ≤ N with the wheel at `psf.filter` and the rest at the calibration filter, in one run. Decisions: one contiguous low segment first (pokes run in index order, so the poke cube should be frequency-ordered); the wheel is moved by `switchSegment`, which sets `m_firstAcq` so the next acquisition settles (the frame in progress spans the move); at the switch a reference is taken before the move (unless one is scheduled there) and one after, so the time-interpolated even part never mixes filters, with no change to `evenCube`; `plannedRefs` allocates them; refuses if the wheel starts at `psf.filter`; warns without PSF blocks with speckles (no scale between segments); execute puts the wheel back after `run()` whatever the outcome. The start PSF block still returns the wheel to the calibration filter before the segment (two extra moves; fwscind is fast, and it keeps `takePsf` self-contained). Outputs: `UNSATLST`, `NREFUNS`, timing column `UNSAT`. Analysis: `load_set` divides planes ≤ UNSATLST by T (mean of the start and end blocks via `psf_flux.run_transmission`) and builds the mean ref from the saturated-filter refs only; checked on try2 with planes 0–499 and refs 0–2 scaled by 0.01 (even recovered to 4e-8). Tests: split at poke 0 (4 refs, 2 unsaturated, even at the core < 200 counts so no mixed refs, `UNSAT` flags, 6 alternating wheel sets), a run ending in the segment restores the wheel, refusal at `psf.filter`.
  - **Core mask on the PSF (user, 2026-11-22, after try6):** at ND1 the bleed stripe (x 145–150, y 138–176) put the saturated pixels' centroid, and so the mask, 9 px from the axis. The 31 px mask still covered the core and stripe, and the mask only gates the linearity check. Decision: after a PSF block with the wheel moved and an unsaturated PSF-filter reference, `recenterCore` centers the mask on that reference: the weighted centroid of the pixels above half maximum over the median background, a symmetric selection, so it is unbiased by the pixel grid. It applies to the rest of the run; the preflight summary still shows the saturated-pixel center, and the start block's `CALMAX` uses it. Header `SATCPSF`. The simulator has `sciBleed` (saturates the pixels past each column's last saturated one); the PSF test runs with a 3 px bleed (saturated center 1.5 px off) and checks `SATCX/SATCY` = 5.5 ± 0.05 and `SATCPSF` = 1.

### 5.8 Follow-ups and edge cases
- **Memory/output size with full-frame camsci1:** all response planes are held in memory, and checkpoints rewrite the full cubes. For a 1024x1024 camsci1 and 2040 pokes, that is about 8.5 GB per cube. The preflight prints the size. If this is too large, use a camsci1 ROI, or change xlinqCal to append planes to the FITS file as it goes.
- **Torn camwfs frames:** the `cnt0` check catches a frame replaced after the post. It cannot catch an in-place writer that is midway through copying the *next* frame before incrementing `cnt0`. Reads happen immediately after each post, so this needs the reader to fall about one frame period behind. The counts of missed and torn frames are in the timing table.
- **DM confirmation:** the wait accepts any `dm01disp` update after the write. The projection is recorded and warned on, but never gates. The settle frames, each ≥ one 250 ms camsci1 frame, cover a `dmcomb` update that was already in progress. If `dmcomb` applies a gain, or the command saturates, the projection will differ from 1 and warn on every poke. Raise `dm.projTol` in that case.
- **Changing camera modes during a run is not supported.** A recreated stream leaves xlinqCal reading the old mapping. Restart and resume with `poke.first`.
- First instrument run (operator): a few pokes with a short `poke.last`. Check `DMPROJ`, the `WFSN` per window (≈ fps ratio × nFrames), `SCIMISS`/`WFSMISS`, and the sign and amplitude of the responses before a full run.
