# Plan: quantized-native MMA FlashAttention for standard Q8_0 K/V

This branch (`beellama-kv-native-mma-kernel`, PR #1 in this fork into
`beellama-kv-cpu-offload-followup`) is both the plan and, from here on, the
implementation branch for this kernel. It stays separate from
`beellama-kv-cpu-offload-followup` so nothing Experiments 001-011 already
confirmed working is put at risk while this is in progress.

Most of this document is still a plan: no kernel code exists yet. But the
open questions that mattered most for starting safely have since been
answered by reading the actual config tables and running the existing test
suite (not just reading code), specifically so the next session can start
Phase 1 directly instead of re-deriving this. See "Resolved before
implementation" below before "Open questions."

## Motivation

Experiment 011 (`docs/cpu-kv-offload-experiments.md`) measured that the CUDA
flash-attention MMA/tile kernel path unconditionally casts K and V to F16
before running, regardless of the persistent cache's storage type or where
`--kv-gpu-layers` places it. On this machine that costs roughly 550-625 MiB of
transient VRAM at 16K-32K context, growing with depth; extrapolated to the
240K context the original review report used, the estimate lands close to
that report's own ~950 MiB figure. Closing it needs the flash-attention kernel
to read quantized K/V directly, the way this fork's KVarN cache already does
for its own type. This plan scopes that work for the standard, non-KVarN
`Q8_0` cache specifically, because `Q8_0`/`Q8_0` is the project's fixed
quality gate (`CLAUDE.md`) and is what Experiments 009-011 were measured
against.

## Scope decision: Q8_0/Q8_0 only, not the full qN_0 matrix

`ggml_cuda_fattn_kv_type_supported` (`fattn.cu:266`) lists eleven quantized
K/V types the vector kernel already accepts. Generalizing the MMA path to all
of them is a substantially bigger undertaking than generalizing it to one.
Restricting the first pass to `Q8_0` K combined with `Q8_0` V:

- matches the only pair this PR's correctness work has actually validated
  (Experiments 009-011's byte-identical checks all used Q8_0/Q8_0);
- is the simplest block format to dequantize (single `ggml_half` scale per
  32-element block, no zero-point, no bit-packing below 8 bits) which lowers
  the risk of a subtle per-lane indexing bug compared to, say, `Q3_0`'s
  packed sub-byte layout;
- leaves the door open to add more types later behind the same hook points,
  once the Q8_0 path is proven correct and its performance is known to be
  worth the added compile time and kernel-selection complexity.

Non-goal: extending KVarN's own descriptor-native kernel. That code
(`fattn-mma-kvarn*.cuh`, `fattn-kvarn-dispatch.*`) already solves this problem
for its own cache type and is out of scope here.

## What is already confirmed, with references

- `block_q8_0` (`ggml/src/ggml-common.h:335-340`) is `{ ggml_half d; int8_t
  qs[32]; }`, 34 bytes. Dequantizing element `i` of a block is
  `float(qs[i]) * __half2float(d)` — no auxiliary tensors, no rotated
  storage, unlike KVarN's three-axis affine format.
- `ggml_cuda_get_best_fattn_kernel` (`fattn.cu:399-579`) already special-cases
  quantized K/V for kernel selection, but only ever chooses between the
  F16-materializing MMA/tile kernels and the native-quantized vector kernel
  (line 508-527); there is no third "quantized MMA" option today.
- `ggml_cuda_flash_attn_ext_get_alloc_size` (`fattn.cu:581-622`) sets
  `need_f16_K = need_f16_V = true` for `BEST_FATTN_KERNEL_TILE` and
  `BEST_FATTN_KERNEL_MMA_F16` unconditionally (`fattn.cu:606-611`). This is
  the allocation that would shrink if the kernel read Q8_0 directly.
- `ggml_cuda_flash_attn_ext_mma_f16` (`fattn.cu:115-244`) is the actual MMA
  runtime entry point. It switches only on `Q->ne[0]` (head dimension) and
  GQA-related shape facts — it never inspects `K->type`/`V->type`. In
  practice it is only ever invoked with the graph's pre-cast F16 tensors; the
  kernel template underneath is generic over `type_K`/`type_V`
  (`fattn-mma-f16.cuh:542`), but nothing currently instantiates or dispatches
  to it with a quantized type except KVarN's sentinel types
  (`GGML_CUDA_FATTN_KVARN_TYPE` / `_ORIGINAL_TYPE`, `fattn-mma-kvarn.cuh:4-5`,
  which are placeholder values above `GGML_TYPE_COUNT`, not real `ggml_type`
  values).
- Inside that template, there are exactly two hook points where the loader
  branches on `type_K`/`type_V`:
  `fattn-mma-f16.cuh:620` (`if constexpr (ggml_cuda_fattn_kvarn_template_type(type_K))`,
  K side) and `fattn-mma-f16.cuh:977` (same pattern, V side). Both branches
  write into a `half2`-typed shared-memory tile (`tile_K`/`tile_V`) with a
  fixed stride (`stride_tile_K`/`stride_tile_V`); everything downstream
  (`load_ldmatrix`, the `mma()` calls) consumes that tile without caring how
  it was populated. This is the actual integration surface: a new branch at
  each hook point that recognizes `type_K == GGML_TYPE_Q8_0` and writes the
  same `half2` tile format, dequantizing from raw Q8_0 bytes instead of
  reading pre-cast F16.
- The non-KVarN `else` branch at each hook point
  (`fattn-mma-f16.cuh:627-635` for K, the V equivalent nearby) calls
  `flash_attn_ext_f16_load_tile`, reading from `K_h2`/`V_h2` — `half2`
  pointers into the pre-cast buffer. A Q8_0 loader would instead take a
  `const char *` raw pointer, exactly like the KVarN branch already does
  (`fattn-mma-f16.cuh:626`, `(const char *) K_h2`), and reinterpret it as
  `const block_q8_0 *`.
- `ggml_kv_tail_attention_merge`/`_segmented` were checked and do not apply
  here (see Experiment 011); they attach extra sources to a single kernel
  launch rather than combining independently executed passes, and are
  specific to KVarN's tail mechanism.
- Experiment 004 already established that changing which backend runs
  attention does not shrink these buffers, ruling out a non-kernel shortcut.

## Resolved before implementation

**Block/tile alignment (was open question 1) — clean, no straddling.**
`ggml_cuda_fattn_mma_get_config_ampere` (`fattn-mma-f16.cuh:60-` onward; RTX
4070 is `cc=890`, which `ampere_mma_available` — `common.cuh:352-354`, checked
before `turing_mma_available` in `ggml_cuda_fattn_mma_get_config`,
`fattn-mma-f16.cuh:240-243` — routes to this table, not the Turing one) gives,
for every `ncols` at `DKQ = DV = 128`:

```
GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128,   8, 128, 2, 128,  64,  64,  64, 2, true);
GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128,  16, 128, 2,  64,  64,  64,  64, 2, true);
GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128,  32, 128, 2,  64,  64,  64,  64, 2, true);
GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128,  64, 128, 2,  64,  64,  64,  64, 2, true);
GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 128, 256, 1,  32,  64,  64,  64, 2, true);
```

(fields: `DKQ, DV, ncols, nthreads, occupancy, nbatch_fa, nbatch_K2, nbatch_V2,
nbatch_combine, nstages_target, Q_in_reg`). `nbatch_K2 = nbatch_V2 = 64` for
every one of them — i.e. `2 * 64 = 128` raw elements loaded per tile batch
along the head-dim axis, which is exactly `128 / 32 = 4` whole Q8_0 blocks.
No lane load straddles a block boundary for our target shape on this
architecture; the loader can assume one or more *whole* blocks per chunk and
does not need general remainder handling for `DKQ = DV = 128` on Ampere/Ada.
(Not re-verified for Turing/Volta/AMD — out of scope per the non-goals below.)

**Phase 0 harness (was "write a new test") — already exists and is already
green.** `tests/test-backend-ops.cpp` already has a `test_flash_attn_ext` case
sweeping `type_KV` including `GGML_TYPE_Q8_0` (`test-backend-ops.cpp:10070,
10106`), and it runs the CUDA backend against a CPU reference automatically.
Confirmed by actually running it (not just reading it) on this branch's build:

```bash
build/bin/test-backend-ops test -o FLASH_ATTN_EXT -b CUDA0 -p "type_K=q8_0"
# 349/349 tests passed, Backend CUDA0: OK
```

This is the Phase 0/4 correctness gate, unchanged — no new test file needed.
The Q8_0 cases already pass today because they go through the existing
F16-cast path; after Phase 2-3, the same command must still pass, now
exercising the new loader, with output expected to be bit-identical (see
"What correct means" below), not just within the existing tolerance.

Unrelated finding from the same run, worth recording but explicitly out of
scope here: running the full, unfiltered `test-backend-ops test -o
FLASH_ATTN_EXT -b CUDA0` (no `-p` filter) segfaults on an `hsk=320, hsv=256,
type_K=f16, type_V=f16` case, in `ggml_cuda_flash_attn_ext_vec`
(`fattn.cu:380`, the `GGML_ABORT("fatal error")` fallthrough when no
`FATTN_VEC_CASE` matches). `ggml_cuda_get_best_fattn_kernel` is choosing the
vector kernel for `D=320`, but `FATTN_VEC_CASES_ALL_D` (`fattn.cu:256-260`)
only instantiates `D ∈ {64, 128, 256, 512}` — 320 isn't one of them. This is
an F16/F16 case, has nothing to do with Q8_0 or this plan, and predates this
branch. Route around it with `-p` when using this test during Q8_0 work;
consider filing it separately.

## Open questions still to resolve while writing kernel code

1. **Warp/lane mapping.** `flash_attn_ext_f16_load_tile`
   (`fattn-mma-f16.cuh:373-452`) assigns KV rows (`i`, the token/context axis)
   across warps via `nwarps`/`threadIdx.y`, and head-dim chunks (`k`) within a
   warp via `threadIdx.x`, using a granularity-halving unroll
   (`ggml_cuda_unroll<6>`) so lanes issue coalesced 16-byte loads of
   contiguous `half2` pairs. That coalescing scheme doesn't transfer directly
   to Q8_0: a block's 32 values need one shared `ggml_half` scale read by
   whichever lanes dequantize that block, which is a different access pattern
   than reading contiguous halves. The `i`-axis assignment (rows across warps)
   can likely be reused as-is; the `k`-axis needs new indexing built around
   "which of the 4 blocks per row, which lane(s) cooperatively dequantize it."
   `flash_attn_ext_kvarn_load_tile` (`fattn-mma-kvarn-impl.cuh:238`) is a
   precedent for a per-block-aware loader targeting the same `half2` tile
   output, but its indexing carries KVarN-specific concerns (rotation,
   staging, tail groups) that don't apply to a plain, unrotated Q8_0 cache and
   should not be copied wholesale — read it for the shape of the problem, not
   as a template to adapt line-by-line.
2. **Runtime dispatch and instantiation growth.** Routing to the new path
   needs: a change to `ggml_cuda_get_best_fattn_kernel` to prefer it over
   MMA_F16-with-cast when `K->type == V->type == GGML_TYPE_Q8_0` and the
   query width exceeds the vector kernel's cutoff; a corresponding
   `need_f16_K = need_f16_V = false` case in
   `ggml_cuda_flash_attn_ext_get_alloc_size`; and new template instantiations
   registered the way `fattn-mma-kvarn-case-decl.cuh` registers KVarN's,
   scoped to the `(DKQ, DV, ncols1, ncols2)` combinations actually reachable
   with Q8_0/Q8_0 (not the full matrix `DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2`
   generates for F16, to avoid a large, mostly-unreachable compile-time
   expansion — mirrors how `CLAUDE.md` already documents the fork's existing
   103-pair vs. 169-pair build-time tradeoff for `GGML_CUDA_FA_ALL_QUANTS`).
   This should sit behind its own build flag, e.g. `GGML_CUDA_FATTN_Q8_NATIVE`,
   defaulting off, following the existing `GGML_CUDA_KVARN` pattern, so it
   does not affect anyone not opting in.
3. **GQA ratio and multi-architecture coverage.** `fattn-mma-f16.cuh`
   branches heavily on `cc` (Turing/Volta/Ada/AMD MFMA/WMMA) and on GQA ratio
   for tile-loading efficiency. This plan's first pass should target this
   machine's architecture (Ada, `sm_89`) only and explicitly punt on
   Volta/AMD paths rather than write untested code for hardware this PR
   cannot validate on.
4. **What "correct" means here.** The current F16-cast path already rounds
   Q8_0 values to `half` before the MMA math runs. A native loader that
   dequantizes Q8_0 straight to the same `half` values feeding the same
   `mma()` calls should therefore be bit-for-bit identical to the current
   path's output, not merely close — that is a strong, checkable invariant
   and should be the correctness bar (same standard as Experiment 010's
   byte-identical greedy-output check), not a tolerance-based comparison.

## Phased plan

**Phase 0 - confirm the baseline, before touching anything.** No new harness
to write (see "Resolved before implementation" above) — just run
`build/bin/test-backend-ops test -o FLASH_ATTN_EXT -b CUDA0 -p "type_K=q8_0"`
on a clean build of this branch and confirm it still says `349/349 tests
passed` (or whatever the current count is if upstream added cases since).
That is the exact command Phase 1 and Phase 4 must keep passing.

**Phase 1 - K/V loader functions.** Write `flash_attn_ext_q8_0_load_tile`
(K and V variants, or one function parameterized like the existing pair) in
a new `fattn-mma-q8.cuh`, reusing the `i`-axis (row/warp) assignment from
`flash_attn_ext_f16_load_tile` and designing new `k`-axis (block/lane)
indexing per open question 1 above. Resolve open question 1 here. Validate
against Phase 0's command with `-p "type_K=q8_0"` still targeting the
existing (unwired) path — this phase does not change dispatch yet, so there
is nothing runtime-visible to test until Phase 2-3; treat this phase as
passing once the loader compiles and a throwaway direct call to it (outside
the dispatcher) produces dequantized values matching `block_q8_0`'s known
scale/value formula on hand-constructed input.

**Phase 2 - hook into `fattn-mma-f16.cuh`.** Add the `type_K ==
GGML_TYPE_Q8_0` / `type_V == GGML_TYPE_Q8_0` branches at the two points
identified above (`fattn-mma-f16.cuh:620` and the V-side equivalent),
producing the same `half2` tile contract the existing branches already
produce, so everything past the load (the `mma()` calls, softmax, output
combine) needs zero changes.

**Phase 3 - instantiation and dispatch.** Add the case/decl files
(`fattn-mma-q8-case.cuh` / `-case-decl.cuh`, mirroring the KVarN pair but for
a plain type instead of a bit-width axis), extend
`ggml_cuda_flash_attn_ext_mma_f16` (or a sibling entry point) to route to
them for `Q8_0`/`Q8_0`, and update `ggml_cuda_get_best_fattn_kernel` and
`ggml_cuda_flash_attn_ext_get_alloc_size` accordingly. Gate all of it behind
`GGML_CUDA_FATTN_Q8_NATIVE` (default off) so the change is inert unless
opted into, matching this fork's existing `GGML_CUDA_KVARN` convention.

**Phase 4 - correctness at the inference level.** Repeat Experiment 010's
correctness protocol: greedy generation (`--temp 0 --seed 1234`, `-no-cnv`)
byte-identical with the flag on vs. off, across a few depths, plus
`ctest -R "test-kvarn|test-adaptive-dm|test-server-loop-guard"` and Phase 0's
`test-backend-ops` command, now exercising the real dispatch path and
expected to stay bit-identical rather than merely within tolerance (open
question 4 below). Do not proceed to measurement until this passes.

**Phase 5 - measurement.** Repeat Experiment 011's `llama-bench --kv-memory`
methodology (compute-buffer bytes vs. depth, vs. `--kv-gpu-layers`) to
confirm the compute buffer actually shrinks and by how much, plus a
prefill/decode throughput sweep to check the new loader doesn't cost more in
compute than it saves in memory bandwidth — an MMA kernel doing per-block
dequantization instead of a flat cast could plausibly be slower per tile
even though it allocates less, and that tradeoff needs to be measured, not
assumed, the same way Experiment 009 measured compression's overhead instead
of assuming it would pay off.

**Phase 6 - decide.** Document the outcome as a new experiment in
`docs/cpu-kv-offload-experiments.md` regardless of which way it goes, keeping
this fork's convention of recording rejections as carefully as acceptances.

## Explicit non-goals for this plan

- Any type other than Q8_0/Q8_0. In particular, not Q4_0/Q4_1 for the
  speculative-draft/MTP/DSpark KV cache: `common_params_speculative_draft`
  (`common.h:343,628`) defaults `cache_type_k`/`cache_type_v` to `GGML_TYPE_F16`,
  and this project's current intent is to keep draft caches at F16. F16 is
  already the kernel's native format, so Experiment 011's finding does not
  apply there at all — no cast, no transient buffer to shrink. Revisit only
  if the draft cache is deliberately moved to a quantized type later; if it
  is, the same hook points apply and Q4_0's block format (packed 4-bit
  nibbles plus scale, no zero-point) is a reasonable second type to add using
  this same design, once Q8_0 is proven.
- Any change to KVarN's own kernel or dispatch.
- Volta, pre-Ada, or AMD code paths.
- Fixing the unrelated `hsk=320` vector-kernel dispatch crash found while
  confirming the Phase 0 baseline (see above). Route around it with `-p`;
  it is not caused by and does not block this plan.
- Anything that touches `--kv-gpu-layers` or the persistent cache placement
  logic from Experiment 010 — this plan is additive to it, not a
  modification of it.
- Any change on the `beellama-kv-cpu-offload-followup` branch itself. This
  plan and its eventual implementation stay on this branch and its PR until
  proven, specifically so Experiments 001-011 stay untouched and mergeable
  independently of how this turns out.

## Honest sizing

This is materially bigger and riskier than anything in Experiments 001-011.
Those were either a wiring change reusing an existing per-layer mechanism
(Experiment 010) or pure measurement (009, 011). This is new CUDA tensor-core
kernel code with a real chance of subtle correctness bugs (lane/lay-out
mistakes in lock-step warp code fail silently more often than they crash),
and it should be treated and reviewed as such — expect this to take
substantially longer than Experiments 001-011 combined, and expect Phase 0-2
to reveal problems this plan could not anticipate from reading the code
alone.

## Starting the next session

Everything below is verified against this branch's actual build, not just
read. Start with Phase 1 directly.

```bash
# confirm the baseline is still what this plan assumes
cmake --build build --config Release --target test-backend-ops --parallel 10
build/bin/test-backend-ops test -o FLASH_ATTN_EXT -b CUDA0 -p "type_K=q8_0"
# expect: 349/349 tests passed (or higher if upstream added cases), Backend CUDA0: OK
```

Files to create: `ggml/src/ggml-cuda/fattn-mma-q8.cuh` (Phase 1 loader).
Files to edit once the loader exists: `fattn-mma-f16.cuh:620` and its V-side
counterpart near line 977 (Phase 2); `fattn.cu` (`ggml_cuda_get_best_fattn_kernel`,
`ggml_cuda_flash_attn_ext_get_alloc_size`, `ggml_cuda_flash_attn_ext_mma_f16`)
plus new `template-instances/fattn-mma-q8-instance-*.cu` files and
`ggml/CMakeLists.txt` (new `GGML_CUDA_FATTN_Q8_NATIVE` option) for Phase 3.

Target shape for the first working instantiation: `DKQ = DV = 128`, Q8_0/Q8_0,
Ampere/Ada only (`ampere_mma_available`), `nbatch_K2 = nbatch_V2 = 64` (4 whole
Q8_0 blocks per tile chunk, no straddling — see "Resolved before
implementation"). Get one `ncols` value working and passing Phase 0's command
before generalizing to the others in the Ampere table above.
