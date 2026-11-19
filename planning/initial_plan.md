
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

### 5.8 Follow-ups and edge cases
- **Memory/output size with full-frame camsci1:** all response planes are held in memory, and checkpoints rewrite the full cubes. For a 1024x1024 camsci1 and 2040 pokes, that is about 8.5 GB per cube. The preflight prints the size. If this is too large, use a camsci1 ROI, or change xlinqCal to append planes to the FITS file as it goes.
- **Torn camwfs frames:** the `cnt0` check catches a frame replaced after the post. It cannot catch an in-place writer that is midway through copying the *next* frame before incrementing `cnt0`. Reads happen immediately after each post, so this needs the reader to fall about one frame period behind. The counts of missed and torn frames are in the timing table.
- **DM confirmation:** the wait accepts any `dm01disp` update after the write. The projection is recorded and warned on, but never gates. The settle frames, each ≥ one 250 ms camsci1 frame, cover a `dmcomb` update that was already in progress. If `dmcomb` applies a gain, or the command saturates, the projection will differ from 1 and warn on every poke. Raise `dm.projTol` in that case.
- **Changing camera modes during a run is not supported.** A recreated stream leaves xlinqCal reading the old mapping. Restart and resume with `poke.first`.
- First instrument run (operator): a few pokes with a short `poke.last`. Check `DMPROJ`, the `WFSN` per window (≈ fps ratio × nFrames), `SCIMISS`/`WFSMISS`, and the sign and amplitude of the responses before a full run.
