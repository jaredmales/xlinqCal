# xlinqCal

Measures synchronized camwfs and camsci1 responses to a sequence of tweeter DM pokes on MagAO-X. The goal is to predict camsci1 images from camwfs images. This is a standalone program: it uses only MILK/ImageStreamIO streams, with no INDI.

## How it works

For each poke shape `k` and each sign in the sequence (`+-`, then `-+`, ... for `seq.nRepeat` pairs; `+--+` by default):

1. Write `initial + sign*amp*shape_k` to `dm.channel` (default `dm01disp04`, written non-passively so `dmcomb` updates even with the loop open). Wait for `dm.comb` (`dm01disp`) to update, and record the projection of its change onto the command as a check.
2. Discard the first camsci1 frame after the poke, then `sci.nSettle` more frames.
3. Average the next `sci.nFrames` camsci1 frames. The camwfs frames between the posts that bound those camsci1 frames are averaged too.

The response for each camera is `(mean(+) - mean(-))/2`, averaged over repeats. It is **not** divided by the amplitude. If the sign doesn't change between consecutive acquisitions (the middle `--`), the DM is not rewritten and no frames are discarded.

## Operator setup (on RTC)

- Loop open, system aligned, cameras running at their intended settings. Don't change camera modes or ROIs during a run.
- camsci1 must appear on RTC. On RTC run `shmimTCPreceive <port>`, then on ICC run `shmimTCPtransmit camsci1 <RTC address> <port>` (see the handbook's shmimTCP page).
- A poke cube: FITS, 50x50xNpoke, in microns, each shape normalized to 1 µm rms.

## Running

```
xlinqCal -c xlinqCal.conf            # see example/xlinqCal.conf
xlinqCal -c xlinqCal.conf --poke.amp=0.05 --poke.first=100 --yes
```

Before poking, xlinqCal checks the streams, measures both frame rates, and prints the estimated run time and output size. It then asks for confirmation (`--yes` skips this). Ctrl-C stops cleanly: the DM channel is restored, completed planes are written, and the log gives the `poke.first` to resume from. The DM channel is also restored on errors and timeouts. If you have a CPU set for non-RT readers, run the program inside it (`taskset`/`cset`).

With the defaults, each poke takes 46 camsci1 frames, about 11.5 s at 4 Hz: 3 DM changes × 2 discarded frames + 4 acquisitions × 10 averaged frames.

## Outputs

In `out.dir`, with prefix `out.prefix` (default `xlinqCal_<UTC>`):

| file | content |
|---|---|
| `<prefix>_camsci1_resp.fits` | camsci1 response cube, one plane per poke |
| `<prefix>_camwfs_resp.fits` | camwfs response cube, one plane per poke |
| `<prefix>_timing.fits` | one row per acquisition: poke, repeat, sign, DM cnt0 and projection, camsci1/camwfs cnt0 ranges, frame counts, missed and torn frames, gate times. Column names are in `TCOLn` |
| `<prefix>.log` | run log |

Headers record the configuration, stream names, datatypes, measured fps, poke range (`POKE0`, `NPOKE`), `COMPLETE`, and start/end times. Partial results are checkpointed every `out.checkpointMin` minutes.

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
