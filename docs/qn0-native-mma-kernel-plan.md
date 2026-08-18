# Plan: quantized-native MMA FlashAttention for standard Q8_0 K/V

This is a design plan, not an implementation. It lives on its own branch
(`beellama-kv-native-mma-plan`, PR into `beellama-kv-cpu-offload-followup`)
so it does not touch anything Experiments 001-011 already confirmed working.
Nothing here has been built or tested; every code reference below is read,
not verified by compiling against.

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

## Open questions to resolve before writing kernel code

1. **Block/tile alignment.** Q8_0 blocks of 32 elements live along the head
   dimension (each 128-wide row is 4 blocks). The load tiles are chunked by
   `nbatch_K2`/`nbatch_V2` (`fattn-mma-f16.cuh:298-311`), which are
   `half2`-element counts along that same axis, config-dependent per
   `(DKQ, DV, ncols)`. Need to confirm `2 * nbatch_K2` and `2 * nbatch_V2` are
   multiples of 32 for every `(DKQ, DV, ncols)` combination this PR would
   enable (at minimum `DKQ = DV = 128`, the shapes used by the models in
   Experiments 001-011); if not, a lane can straddle a block boundary and the
   loader needs to handle that instead of assuming one block per load.
2. **Warp/lane mapping.** `flash_attn_ext_f16_load_tile`'s existing lane
   assignment is built around loading contiguous `half2` pairs; a Q8_0
   loader needs each lane to know which block it is in (for the scale) and
   which int8 pair within that block (for the value), which is a different
   indexing shape. `flash_attn_ext_kvarn_load_tile`
   (`fattn-mma-kvarn-load.cuh`) is the precedent to study for this, even
   though KVarN's own indexing is more complex than what Q8_0 needs (no
   rotation, no staging, no tail groups — those exist for KVarN's streaming
   design and would all be dead weight here).
3. **Runtime dispatch and instantiation growth.** Routing to the new path
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
4. **GQA ratio and multi-architecture coverage.** `fattn-mma-f16.cuh`
   branches heavily on `cc` (Turing/Volta/Ada/AMD MFMA/WMMA) and on GQA ratio
   for tile-loading efficiency. This plan's first pass should target this
   machine's architecture (Ada, `sm_89`) only and explicitly punt on
   Volta/AMD paths rather than write untested code for hardware this PR
   cannot validate on.
5. **What "correct" means here.** The current F16-cast path already rounds
   Q8_0 values to `half` before the MMA math runs. A native loader that
   dequantizes Q8_0 straight to the same `half` values feeding the same
   `mma()` calls should therefore be bit-for-bit identical to the current
   path's output, not merely close — that is a strong, checkable invariant
   and should be the correctness bar (same standard as Experiment 010's
   byte-identical greedy-output check), not a tolerance-based comparison.

## Phased plan

**Phase 0 - isolated correctness harness, before touching the dispatcher.**
Write a standalone test (extending `test-kvarn`'s pattern or a new
`test-fattn-q8-native`) that builds small, fixed-shape `GGML_OP_FLASH_ATTN_EXT`
graphs with random Q8_0 K/V, runs both the existing cast-based path and the
new loader in isolation, and diffs outputs exactly. Getting this passing on
synthetic tiles, without wiring the new path into the runtime kernel
selector, is the checkpoint before anything touches the code path real
inference traffic goes through.

**Phase 1 - K/V loader functions.** Write `flash_attn_ext_q8_0_load_tile`
(K and V variants, or one function parameterized like the existing pair) in
a new `fattn-mma-q8.cuh`, following `flash_attn_ext_kvarn_load_tile`'s
signature shape but with Q8_0's much simpler math and no staging/rotation.
Resolve open question 1 and 2 here. Validate against Phase 0's harness.

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
`ctest -R "test-kvarn|test-adaptive-dm|test-server-loop-guard"` and the new
Phase 0 test. Do not proceed to measurement until this passes.

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

- Any type other than Q8_0/Q8_0.
- Any change to KVarN's own kernel or dispatch.
- Volta, pre-Ada, or AMD code paths.
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
