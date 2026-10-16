#pragma once

#include "common.cuh"
#include "ggml-backend-impl.h"

#include <cstddef>

// AllReduce of two GPUs without peer access, through pinned host memory with the copy engines.
//
// Each GPU converts its F32 contribution to BF16, copies it to host memory in chunks and copies the chunks of the
// other GPU back as they arrive, so both directions of the PCIe links are in use at the same time. No SM is busy
// during the transfer, unlike NCCL, whose kernels hold an SM each and push a kernel that has one block per SM into a
// second wave. The GPUs are ordered with events only, the host never waits.
struct ggml_cuda_ce_ar;

// the buffers (16 MiB VRAM and 8 MiB pinned memory per GPU) are allocated with the first reduction
// peer: the GPUs have peer access, each copies its contribution straight into the other's memory instead of
// through host memory (no pinned memory then)
// returns nullptr on MUSA, which lacks the pinned-memory API used here
ggml_cuda_ce_ar * ggml_cuda_ce_ar_init(const int * devices, size_t n_devices, bool peer = false);

void ggml_cuda_ce_ar_free(ggml_cuda_ce_ar * ar);

// in-place sum of tensors[0] and tensors[1], contiguous F32 of the same size on the devices of backends[0] and [1]
// returns false without enqueuing anything if the tensors are not supported or the buffers cannot be allocated
bool ggml_cuda_ce_ar_allreduce(ggml_cuda_ce_ar * ar, ggml_backend_t * backends, ggml_tensor ** tensors);
