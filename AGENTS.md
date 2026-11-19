# xlinqCal Agent Notes

xlinqCal is a standalone MagAO-X calibration tool. The MagAOX repository (`magao-x/MagAOX`, `dev` branch) is the interface specification, and its `AGENTS.md` rules apply here, including:

- Rule 0: agents never control the instrument. Develop and test only against `xlinqSim` streams on a workstation (check `hostname` is not `exao*`). The operator runs xlinqCal on RTC.
- Code style, Doxygen, and naming rules (1-13, 16, 26), header-only classes with `main`-only `.cpp` files (21), and `clang-format` (`make format`), then restore `{ //mutex scope`.
- Real-time stream rules (28): camwfs is a loop stream, so readers wait on semaphores (no spinning), copy each frame once, and keep per-frame work vectorized.
- Commit discipline (14, 15, 22, 23): `<username>/<feature>` branches, small commits, brief messages, plan files updated with engineering decisions.

Local notes:

- Simulated stream names must start with `xlsim_` (enforced by `xlinqSim`), so tests can never create or overwrite instrument streams.
- Run `make test` after any change. The integration tests run xlinqCal in-process against `xlinqSim`.
- Engineering decisions and execution notes are in `planning/initial_plan.md`.
