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

In `out.dir`, with prefix `out.prefix` (default `xlinqCal_<UTC>`):

| file | content |
|---|---|
| `<prefix>_<cam>_resp.fits` | odd part `(mean(+) − mean(−))/2`, one plane per poke |
| `<prefix>_<cam>_mean.fits` | symmetric mean `(mean(+) + mean(−))/2`, one plane per poke |
| `<prefix>_<cam>_even.fits` | even part `mean − ref` (ref interpolated in time), one plane per poke. `EVEN1SID` counts planes that used only the nearest reference (e.g. after a stop) |
| `<prefix>_<cam>_ref.fits` | zero-poke reference frames, in acquisition order |
| `<prefix>_timing.fits` | one row per acquisition: poke, repeat, sign (0 = reference, with the reference index in REPEAT), DM cnt0 and projection, camsci1 cnt0 range and frame counts, gate times, then for each WFS stream `j` its cnt0 range, frame count, and missed and torn frames (`WjCNT0A`, `WjCNT0B`, `WjN`, `WjMISS`, `WjTORN`). Column names are in `TCOLn` |
| `<prefix>.log` | run log |

`<cam>` is the stream name (`camsci1`, and each WFS stream). Headers record the configuration, stream names, datatypes, measured fps, poke range (`POKE0`, `NPOKE`), references (`REFEVERY`, `NREF`), `PRODUCT`, `COMPLETE`, and start/end times. Partial results are checkpointed every `out.checkpointMin` minutes.

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
