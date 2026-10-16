# Compute buffers that follow the batches

By default a context reserves its compute buffers once, for the worst case: a full ubatch whose attention reads the whole KV cache. Two options size them by the batches instead. Both are off by default, and the outputs do not change.

| Option | What it does |
|---|---|
| `--phase-aware-workspace` | Reserves only for the generation batches (`n_outputs_max` or one token per sequence) while there are no larger ones, and for `n_ubatch` when a prompt comes. An MTP draft context (`--spec-type draft-mtp`) computes in the compute buffers of its target context instead of its own. |
| `--live-context-workspace` | Sizes the attention part for the used part of the KV cache instead of the whole cache. The steps are powers of two from 1/8 of the cache (at least 256 cells), so a growing context reserves again at most three times. |

A smaller reservation takes effect only after 64 calls in a row that need less. Short turns between long ones, or the generation between two prompts, then do not allocate the buffers twice. A larger one takes effect at once. When no server slot is busy, the server also returns the memory of the CUDA memory pools (`llama_trim_transient_memory()`).

The memory these options free is a saving for the time it is not needed, not a smaller worst case. A context that reaches its full size needs the full buffers again, so that memory must still be free then. If it is not, `llama_decode()` returns -2.

## Sharing with the MTP draft context

The target and the draft context compute one after the other: the draft needs the hidden states of the target. With `--phase-aware-workspace` the draft context therefore borrows the compute buffers of the target, and the shared buffers have the size of the larger of the two.

- Before either context uses the buffers, it waits until the other has finished with them.
- With `--split-overlap`, the draft borrows the buffers of lane 0. Lane 1 keeps its own buffers, because it computes while lane 0 does.
- The buffers of a meta backend (`--split-mode tensor`) are shared through an alias with its own tensor mapping.

## Measured

Qwen3.8-27B UD-Q4_K_S with its MTP head, on an RTX 4070 + RTX 3060.

- Settings: `-sm tensor -ts 0.625,0.375 -as 1,3 --split-overlap -ub 256`, 2 slots of 131072 tokens, q5_1 K/V cache.
- Requests: greedy, run through `llama-server`.

| | default | `--phase-aware-workspace` | both options |
|---|--:|--:|--:|
| VRAM after load, 4070 / 3060 (MiB) | 11309 / 11497 | 10985 / 11173 | 10985 / 11173 |
| VRAM peak, 1k prompt (MiB) | 11421 / 11603 | 11321 / 11503 | 11209 / 11391 |
| VRAM peak, 10k prompt (MiB) | 11509 / 11691 | 11403 / 11583 | 11291 / 11471 |
| VRAM peak, 95k prompt (MiB) | 11513 / 11695 | 11407 / 11589 | 11431 / 11639 |
| prompt processing, 10k (t/s) | 991 | 972 | 973 |
| prompt processing, 95k (t/s) | 733 | 728 | 723 |
| generation, 10k (t/s) | 44.9 | 44.9 | 44.7 |

- **Outputs:** all single-slot requests give the same text in every column. Two slots that compute at the same time can differ between runs, also without the options, because the batches then depend on when the requests arrive.
- **Cost:** each time the buffers grow, they are allocated again, about 160 ms for both GPUs. The waits between the target and the draft cost about 1% of the prompt processing.
