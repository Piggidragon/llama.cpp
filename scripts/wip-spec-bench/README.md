# WIP: speculative decoding benchmarks (parked)

Tools behind the measurements of branch `spec/adaptive-mtp-length` (Qwen3.8-27B, RTX 4070 + RTX 3060, production preset).

- `run.sh TAG REPS TASKS PAR key=value ...`: test server on :18900 from `base.ini` with preset overrides, `bench.py` against it, trace in `trace-TAG.txt` (`LLAMA_SPEC_TRACE`)
- `prod.sh unload|load`: production model in the router on :5803
- `direct.sh LOG [args]`: the child server with the production args of `child-args.json`, for libllama logs (`-lv 4`)
- `ana.py` (acceptance by draft position and probability), `sim.py` (offline draft length policies), `cyc.py`, `shape.py` (step time after a batch shape change), `sweep.py` (draft max sweep)
- `results.jsonl`: all runs (tag, task, t/s, drafted and accepted tokens)

Results (2026-11-03): with `--spec-rollback 2` (today's VRAM) MTP depth 2 is the fastest also for code; deeper drafts lose to the replays of rejections past the rollback rows. `--graph-cache` removes the ~18 ms of a batch shape change under split mode tensor, but each cached shape keeps ~26 MiB per GPU of CUDA graphs.
