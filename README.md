# What this fork adds

Branch `master` mirrors [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). Branch `main` is upstream plus the features below. Each one is off until its flag enables it, except the changes to the server's RAM prompt cache, which are always on (see the prompt cache item). The rest of this file is the upstream README.

## Features

- **Quantized-native FlashAttention (CUDA).** The FlashAttention kernel reads a quantized K/V cache in place instead of first copying the attention window to F16.
  - Benefit: faster generation at long context (Qwen3.8-27B: +6-11% at 16k, up to +39% at 64k), and no F16 copy in the compute buffer (1.6 GiB less at 262k context).
  - Enable: `--flash-attn-native-quants` (`-fanq 1` in llama-bench) with a quantized cache, the same type for K and V (`-ctk q5_1 -ctv q5_1`). `q4_1`, `q5_0` and `q5_1` need `-DGGML_CUDA_FA_QUANTS=<pair>` or `all` at build time. [Details](docs/quantized-native-flash-attention.md)
- **Prompt cache on disk.** A disk level below the RAM prompt cache. Prompts are stored as KV blocks keyed by their prefix, so conversations with a shared prefix share blocks, and a growing conversation writes only its new blocks.
  - Benefit: conversations survive RAM eviction, restarts and model swaps, and resume without processing the prompt again. The next turn of a 25k-token conversation writes 0.16 GB instead of 2.3 GB.
  - Enable: `--cache-disk PATH` (server), sized with `--cache-disk-size` (MiB). [Details](tools/server/README.md#prompt-cache-on-disk)
  - Always on, with the RAM prompt cache (`--cache-ram`, on by default):
    - The RAM cache stores prompts as KV blocks of `--cache-block-size` tokens (default 2048).
    - A cached prompt keeps only its newest 3 context checkpoints.
    - The server restores a cached prompt into a slot when that prompt shares at least 256 more tokens with the request than the slot does.
    - SIGINT and SIGTERM are handled on their own thread.
- **Attention split** (`--split-mode tensor`). The attention heads and their KV cache get their own split across the GPUs, separate from `--tensor-split`.
  - Benefit: with GPUs of different speed, the faster one takes more of the weights while the other one holds more of the KV cache.
  - Enable: `--attn-split 1,3` (`-as`), next to e.g. `--tensor-split 0.625,0.375`. [Details](docs/multi-gpu.md)
- **AllReduce by size** (`--split-mode tensor`, two CUDA GPUs). Small reductions go through a kernel with a handshake instead of NCCL, and large ones through the copy engines. With `p2p`, the GPUs write directly into each other's memory.
  - Benefit: lower latency per decode step, and prompt processing that keeps every SM free for compute.
  - Enable: `--allreduce host` (no peer access needed) or `--allreduce p2p` (needs peer access between the GPUs; falls back to `host` without it). [Details](docs/multi-gpu.md#7-two-gpus-over-pcie---allreduce)
- **Split overlap** (`--split-mode tensor`). Large ubatches alternate between two schedulers, so the AllReduce of one ubatch overlaps with the compute of the next.
  - Benefit: faster prompt processing when the link between the GPUs is slow (Qwen3.8-27B on two PCIe GPUs: 733 -> 947 t/s at 10k context).
  - Enable: `--split-overlap`, with half the usual `--ubatch-size` (each scheduler has its own compute buffers). Best together with `--allreduce host` or `p2p`. [Details](docs/multi-gpu.md#8-overlapping-the-allreduce-with-compute---split-overlap)

## Upstream vs this fork

Setup: RTX 4070 (12 GB, PCIe 4.0 x16) + RTX 3060 (12 GB, PCIe 3.0 x4), Qwen3.8-27B UD-Q4_K_S with its MTP head (2 draft tokens), `llama-server` with 2 slots of 131k context, q5_1 K/V cache.

- Upstream: `-sm tensor -fa on -ctk q5_1 -ctv q5_1`, the best split that fits.
- This fork: additionally `--flash-attn-native-quants -ts 0.625,0.375 --attn-split 1,3 --allreduce p2p --split-overlap -ub 256 --cache-disk ...`.

| | upstream | this fork |
|---|--:|--:|
| prompt processing, 10k context (t/s) | | |
| prompt processing, 95k context (t/s) | | |
| generation, 10k context (t/s) | | |
| generation, 95k context (t/s) | | |
| resume a 25k-token conversation after a restart (s) | | |
| VRAM after a 95k request, 4070 / 3060 (MiB) | | |

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
