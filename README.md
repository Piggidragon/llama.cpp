# What this fork adds

Branch `master` mirrors [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). Branch `main` is upstream plus the features below. A feature with a flag is off until the flag enables it; the rows marked *always on* and the RAM prompt cache changes in the notes under the table need nothing. The rest of this file is the upstream README.

## Features

All measurements: Qwen3.8-27B. The last three rows need `--split-mode tensor`; the AllReduce paths need two CUDA GPUs.

| Feature | What it does | Benefit | Enable |
|---|---|---|---|
| Quantized-native FlashAttention (CUDA) | Reads a quantized K/V cache in place instead of copying the attention window to F16 | Generation +6-11% at 16k context, up to +39% at 64k; 1.6 GiB less compute buffer at 262k | `--flash-attn-native-quants` (`-fanq 1`), same quantized type for K and V. [Details](docs/quantized-native-flash-attention.md) |
| Prompt cache on disk | Stores prompts as KV blocks, keyed by their prefix, on disk below the RAM cache | Conversations survive RAM eviction, restarts and model swaps; a growing conversation writes only its new blocks (0.16 GB instead of 2.3 GB at 25k tokens) | `--cache-disk PATH`, `--cache-disk-size` (MiB). [Details](tools/server/README.md#prompt-cache-on-disk) |
| Conversation cache for hybrid models (root checkpoint and spread eviction after [Strata](https://github.com/Niko1221/Strata) by [@Niko1221](https://github.com/Niko1221)) | Keeps a checkpoint at the end of the system prompt for good and caches it as an entry of its own (also on disk); thins out old checkpoints geometrically instead of dropping the oldest; saves a conversation before a new chat with the same system prompt takes its slot | A new chat with a 9k-token system prompt: 15.4 -> 0.3 s in the other slot, 9.9 -> 0.4 s after a restart; a conversation back after another chat took its slot 3.9 -> 0.3 s (Qwen3.8 27B) | `--checkpoint-root 2048`, `--checkpoint-spread`, `--cache-save-min 512`. [Details](tools/server/README.md#conversation-cache-for-hybrid-and-recurrent-models) |
| Compute buffers that follow the batches (by [@GenerelSchwerz](https://github.com/GenerelSchwerz)) | Sizes the compute buffers by the current batch; an MTP draft context uses its target's buffers | Up to 212 MiB less VRAM per GPU (106 MiB at full context), prompt processing 1-2% slower | `--phase-aware-workspace`, `--live-context-workspace`. [Details](docs/compute-workspace.md) |
| MTP draft vocabulary (after the draft vocabulary of [Strata](https://github.com/Niko1221/Strata) by [@Niko1221](https://github.com/Niko1221)) | The MTP draft head projects onto a list of frequent tokens and its sampler goes over those only; the target still checks every draft against the whole vocabulary. At load the rows of the output head are reordered so the list's tokens come first on each GPU, and the draft head is a view of those rows (a copy where the head is not a tensor of its own on a device); the target's logits are put back in token order on the CPU when they are read. `llama-draft-vocab` builds the list from a text corpus | Qwen3.8 27B, MTP depth 2, a list of 66k of the 248k tokens (English, German, code, man pages): generation +7% at temperature 1.0 and +6% greedy, in German too (four English/code tasks 59.2 -> 63.6 t/s, code edit 67.5 -> 72.8, German letter 47.2 -> 50.0); no extra VRAM (a copy of the rows took 168 / 100 MiB on the two GPUs and did not fit next to a 95k-token prompt at 262k context) | `--spec-draft-vocab FILE`. [Details](tools/draft-vocab/README.md) |
| Choice between drafters (after the choice between prompt lookup and MTP in [Strata](https://github.com/Niko1221/Strata) by [@Niko1221](https://github.com/Niko1221)) | With several `--spec-type`, each step takes the draft with the most expected tokens per time: an n-gram draft copied from the context, the MTP draft, or none. The acceptance of each draft position is learned per slot, also from the n-gram drafts that were not used and from the part of a draft cut off; the step times per number of verified tokens (a recurrent or hybrid model takes one pass over the model per draft length in a batch). No conversation is made more than 10% slower by another's draft | Qwen3.8 27B, one slot, `ngram-simple,draft-mtp`: editing a quoted docker-compose file 78 -> 123 t/s at temperature 1.0, 79 -> 125 greedy (median of 4 requests in a row: the acceptance is learned over them); code edit, German letter and English answer as with MTP alone, within 2% (n-gram first instead: compose 124 t/s, but code edit -9% / -15%). Two slots generating at once: as MTP alone | `--spec-type ngram-simple,draft-mtp --spec-select` |
| Top-k of the logits on the GPUs | Each GPU finds the largest logits of its share of the vocabulary with a small graph after the decode, and only those come to the host: as many as the request's sampler needs (its top-k plus the tokens penalties and negative logit biases can lower, or 1 for greedy). The CPU samplers run over them instead of the whole vocabulary. Exact: only for such sampler chains (penalties that raise logits, e.g. a negative frequency penalty, take all logits); a token forced by the reasoning budget joins the candidates; while a grammar constrains the tokens (e.g. a tool call after its trigger) the slot takes all logits, and a draft verification stops at the token that starts the grammar | Qwen3.8 27B, 4070 + 3060, MTP + `--spec-select`: docker-compose edit 115 -> 131-133 t/s at temperature 1.0 (49-token verifies: 5.5 ms less copy over the 3060's PCIe x4), mean of four tasks +8%; outputs bitwise identical | `--logits-topk` |
| Faster generation steps | MMVQ launch parameters for RTX 30 and 40 cards (compute capability 8.6, 8.9); a penalties sampler that skips its hash map; one thread per element for short non-contiguous concats; the scheduler queues the copies of a graph's inputs and waits once per split instead of once per input and GPU | Generation with MTP 63.5 -> 68.8 t/s; matmuls of 2-8 tokens (MTP verify) up to +17%; with a repetition penalty, 845 -> 1.6 us of sampling per token; queued input copies +1-2% with MTP under `-sm tensor` (a PCIe x4 GPU waited ~30 us per input) | Always on |
| Faster context checkpoints | Hybrid and recurrent models: a slot shares its checkpoints with the RAM prompt cache instead of copying them, and state copies run through pinned memory on all GPUs at once | A request with a 1k-token prompt 1.51 -> 1.23 s; a 150 MB checkpoint copies in 28 ms | Always on |
| Speculative checkpoints on the GPU | Hybrid and recurrent models: the checkpoint that the verification of a long draft needs (restored when part of the draft is rejected) stays in a device snapshot of the recurrent state, split across the GPUs like the state, instead of going to host memory and back: a copy within each GPU instead of two over PCIe (~20 ms each over the 3060's PCIe 3.0 x4 for Qwen3.8 27B). One slot uses the snapshot at a time, the others keep their checkpoints in host memory | `--spec-select` with `ngram-simple,draft-mtp`: editing a quoted docker-compose file 119 -> 122 t/s at temperature 1.0, 119 -> 128 greedy (the chooser takes more n-gram drafts); code edit, German letter and English answer within 2%. N-gram drafts first (without `--spec-select`): compose 107 -> 123 t/s. Outputs unchanged. Costs the recurrent state of one sequence in VRAM: 94 + 58 MiB | `--spec-ckpt-snapshot` |
| Prefill budget | While a slot generates, the prompts of the other slots go in steps of N tokens instead of `--batch-size` | Generation next to a 20k-token prompt 1.7 -> 11 t/s (N = 64) or 14 t/s (N = 32); that prompt 841 -> 393 or 245 t/s | `--prefill-budget N` |
| Concurrent prefill | While slots generate, a second context on the same KV cache and recurrent state computes the prompts of the other slots at the same time, one after the other (the shortest first); the generating slots run on high-priority streams | Generation next to a 20k-token prompt 1.7 -> 21 t/s with the prompt at 513 t/s (two PCIe GPUs, MTP); with 3 slots 2 -> 22 t/s next to two prompts; also for qwen4exp; ~170 MiB more VRAM per GPU at deep context | `--concurrent-prefill` (`--parallel` >= 2; with `--kv-unified`, also `--kv-unified-gather`, and no sliding-window attention) |
| Compact attention masks (after the compact causal masks by [@Anbeeld](https://github.com/Anbeeld) in [@GenerelSchwerz](https://github.com/GenerelSchwerz)'s fork, [#53](https://github.com/GenerelSchwerz/llama.cpp/pull/53)) | Where every token of a ubatch sees a prefix of the KV cells (a causal mask over one sequence per stream, no SWA or ALiBi; a KV cache per slot or of one sequence, a unified cache of several gets the same ranges with `--kv-unified-gather`), the attention mask is built on the GPU from one range per token instead of being built on the CPU and uploaded | Prompt processing at 93k context 716-730 -> 751-756 t/s (the mask is 48 MB per ubatch there, and as much for the MTP draft); no extra VRAM | `--compact-attn-mask` |
| Unified KV cache without the cost | With `--kv-unified` the attention reads only the cells of the ubatch's own sequences, through a list of them (masked by a range per token where possible), so a slot does not pay for the context the other slots hold; idle slots keep their cells until the cache needs the room; concurrent prefill works with it | Next to a slot holding 93k tokens: generation 42.0 -> 55.9 t/s, a 19.7k-token prompt 518 -> 965 t/s (per-slot KV cache: 57.8, 971); one conversation can take the whole context | `--kv-unified --kv-unified-gather` (FlashAttention on CPU or CUDA, head size 128 or 256) |
| Attention split | Splits the attention heads and their KV cache separately from `--tensor-split` | The faster GPU takes more weights, the other one more KV cache | `--attn-split 1,3` (`-as`). [Details](docs/multi-gpu.md) |
| AllReduce by size | Small reductions through a kernel with a handshake, large ones through the copy engines, instead of NCCL | Prompt processing 866 -> 1181 t/s, generation 34.2 -> 37.6 t/s (two PCIe GPUs, with split overlap) | `--allreduce host`, or `p2p` with peer access. [Details](docs/multi-gpu.md#7-two-gpus-over-pcie---allreduce) |
| Split overlap | Large ubatches alternate between two schedulers, so the AllReduce of one overlaps with the compute of the next | Prompt processing 733 -> 947 t/s at 10k context (two PCIe GPUs) | `--split-overlap` with half the usual `--ubatch-size`. [Details](docs/multi-gpu.md#8-overlapping-the-allreduce-with-compute---split-overlap) |

Notes:
- **Build:** `q4_1`, `q5_0` and `q5_1` for the native FlashAttention need `-DGGML_CUDA_FA_QUANTS=<pair>` (or `all`).
- **Always on, without a flag:** with the RAM prompt cache (`--cache-ram`, on by default), the server
  - stores prompts as KV blocks of `--cache-block-size` tokens;
  - keeps the newest 3 context checkpoints of a cached prompt;
  - restores a cached prompt into a slot when it shares at least 256 more tokens with the request than the slot does;
  - handles SIGINT and SIGTERM on their own thread.

## Upstream vs this fork

Measured on 2026-10-22: RTX 4070 (12 GB, PCIe 4.0 x16) + RTX 3060 (12 GB, PCIe 3.0 x4), peer-to-peer access through patched kernel modules and libcuda for both servers, Qwen3.8-27B UD-Q4_K_S with MTP depth 2, `llama-server` with 2 slots and `--split-mode tensor`, q5_1 K/V cache and q4_0 draft K/V cache on both.

- **This fork** (`main` 3fc7684fa) runs the production config: 2 x 131k context, `-ts 0.625,0.375 --attn-split 1,3 -ub 256 --flash-attn-native-quants --allreduce p2p --split-overlap --phase-aware-workspace --live-context-workspace --concurrent-prefill --compact-attn-mask --spec-type ngram-simple,draft-mtp --spec-select --spec-draft-vocab <English + German list> --logits-topk --spec-ckpt-snapshot --cache-disk ... --checkpoint-root 2048 --checkpoint-spread --cache-save-min 512`.
- **Upstream** (ggml-org `master` a37cc4610, built with all FlashAttention quants, `GGML_CUDA_FA_ALL_QUANTS=ON`) runs the largest context tested that leaves about 400 MiB free on each GPU after loading: 2 x 82k at `-ts 0.5,0.5 -ub 512`.
  - At 2 x 131k it runs out of memory at `-ts 0.5,0.5`, and with the smaller q5_0 cache at every split tried from 0.4375 to 0.625. The compute buffer, 1.0-1.25 GiB on the 4070, holds the F16 copy of the attention window that `--flash-attn-native-quants` avoids.
  - 2 x 86k loads with 201 / 278 MiB free, 2 x 90k with 29 / 106.
  - The fork's split, `-ts 0.625,0.375`, puts more weights on the faster 4070, which helps generation. Upstream cannot use it: it runs out of memory there even at 2 x 41k. The fork can, because `--attn-split 1,3` moves the KV cache to the 3060 and its FlashAttention needs no F16 copy.
- **The 95k values** (marked †) come from a second upstream config, 2 x 98k. At `-ub 512` it loads only with an F16 draft cache: q4_0 and q8_0 do not fit in the MTP draft context. At 1k-64k it is within 7% of the 2 x 82k config. With `-ub 256`, 2 x 106k still loads, with 73 / 150 MiB free (not measured).
- **Drafting:** upstream drafts with MTP alone (it has no `--spec-select`). Its chain `--spec-type ngram-simple,draft-mtp` always takes an n-gram draft when there is one; it was measured for the task table only.

Method:
- **Sampling:** both servers get the fork's production defaults on the command line: temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5, reasoning effort medium. The client sends no sampling parameters.
- **CPU check:** before every request the client waited until other CPU use stayed below 1.5 cores. That is the machine's CPU time minus the server's and the client's, so kernel threads count too. A request during which it went higher was repeated once; a request disturbed on both attempts was left out.
- **VRAM:** sampled by nvidia-smi every 250 ms.
- **Prompts:** the tasks are English and German, the languages of the draft vocabulary.
- The feature rows above measure one feature at a time with other prompts, so their numbers differ from these.

**Generation by task** (generated tokens per second, thinking included; median of 3 requests after a warm-up round):

| task | this fork (n-gram or MTP, `--spec-select`) | upstream (MTP) | upstream (n-gram, then MTP) |
|---|--:|--:|--:|
| reasoning (probability puzzle) | 76.8 | 56.4 | 52.6 |
| creative writing, English | 54.5 | 39.8 | 40.0 |
| creative writing, German | 53.8 | 40.2 | 39.6 |
| writing code (Python module + tests) | 73.0 | 52.2 | 49.1 |
| editing code (C++ functions) | 71.5 | 53.5 | 56.9 |
| editing a quoted docker-compose file | 193.5 | 58.2 | 105.0 |
| explaining (B-tree vs binary search tree) | 61.0 | 43.4 | 44.3 |

Notes:
- **Different outputs:** each request writes its own text, and the thinking length varies. For example, code edit came out at 0.8k-1.7k tokens, and one fork reasoning request thought 2.8k tokens and ran at 66.3 t/s. Writing code stops at the 4096-token limit in every request.
- **First fork run:** a first fork run, with other CPU use of up to 1 core, measured within 2% of these values, except code edit (76.3 t/s, within that task's spread) and the compose edit (145-154 t/s). The compose edit's fast steps depend more on the CPU.

**Prompt processing and generation by context** (a German Wikipedia text to summarize, thinking off, 384 tokens generated):

| prompt | prompt processing (t/s), fork / upstream | generation at that depth (t/s), fork / upstream |
|---|--:|--:|
| 1k | 643 / 576 | 64.0 / 48.4 |
| 4k | 933 / 659 | 65.3 / 45.8 |
| 10k | 833 / 668 | 52.0 / 43.9 |
| 32k | 924 / 627 | 59.4 / 38.4 |
| 64k | 786 / 565 | 50.4 / 32.3 |
| 95k (96.7k tokens) | 713 / 518 † | 36.7 / 26.9 † |

Notes:
- **Samples:** up to 10k, both columns are the median of 2 requests with different texts. Single requests vary a lot, e.g. fork generation at 10k 43.2 and 60.9 t/s, upstream at 1k 43.8 and 53.0.
- **From 32k:** prompt processing is one request. Generation is the median of that request and 2 repeats of it from the cache.
- **CPU:** the fork's run had more other CPU use than upstream's (median 0.8 against 0.1 cores).

**Two slots, and a long prompt next to a generating slot:**

| | this fork | upstream (MTP) |
|---|--:|--:|
| two slots generating at once, combined t/s: German story + Python module / B-tree explanation + English story / probability puzzle + compose edit | 96.0 / 92.2 / 127.1 | 59.3 / 60.1 / 78.0 |
| generating slot while the other slot processes a 20k / 64k / 95k prompt (t/s) | 22.3 / 20.1 / 18.9 | 1.3 / 1.2 / 1.1 † |
| that prompt meanwhile (t/s) | 542 / 489 / 457 | 622 / 530 / 477 † |
| first token of a 26k-token conversation, cold / after a server restart (s) | 31.6 / 1.3 | 44.4 / 42.0 |
| stopping the server (s) | 11.0 | 2.3 |

Notes:
- **Two slots:** each value is the median of 2 runs per pair. Combined t/s counts the tokens of both requests in the time both were streaming: the token counts come from the server's per-token count, the time from when both streams had sent their first token until the first one ended. In the third pair upstream drafts with MTP only; its n-gram chain, not run here, edits the compose file at 105 t/s on one slot.
- **Long prompt next to a generating slot:** the fork computes the prompt in a second context next to the generating slots (`--concurrent-prefill`). In exchange its prompt goes 4-13% slower. Upstream processes the prompt in the generation batches, which almost stalls the generating slot until the prompt is done.
- **Restart:** after a restart the fork restores the conversation from its disk prompt cache: 37 tokens computed, 26,485 read from the cache. Upstream computes all of it again; it has no disk prompt cache, only manual slot save and restore. The fork's stop takes longer because it writes its prompt cache to disk on exit.
- **Load time:** both load in 5-6 s with the model in the page cache; starts without it took 14-19 s.

**Memory:**

| | this fork | upstream (MTP) |
|---|--:|--:|
| context | 2 x 131k | 2 x 82k (2 x 98k †) |
| free VRAM after loading, 4070 / 3060 (MiB) | 581 / 436 | 407 / 480 (221 / 294 †) |
| least free VRAM during the runs, 4070 / 3060 (MiB) | 156 / 52 | 335 / 416 (151 / 234 †) |

Notes:
- **Fork:** its compute buffers follow the batch and the context, so its free VRAM falls during long prompts. The least, 52 MiB on the 3060, came with a 95k prompt next to a generating slot. The deepest prompt measured was 96.7k tokens, not the full 131k.
- **2 x 98k config:** it ran only the prompt and next-to tests.

---

# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

```bash
# curl
curl -LsSf https://llama.app/install.sh | sh

# powershell
irm https://llama.app/install.ps1 | iex
```

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
