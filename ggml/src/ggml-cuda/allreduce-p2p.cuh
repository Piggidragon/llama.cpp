#pragma once

#include "common.cuh"
#include "ggml-backend-impl.h"

#include <cstddef>

// AllReduce of two GPUs with peer access for small reductions (decode): one kernel per GPU writes its contribution
// and then a flag straight into the memory of the other GPU and waits for the flag the other one writes into its
// own memory. Only local memory is polled and the data crosses the link once, as posted writes.
struct ggml_cuda_p2p_ar;

// enables peer access between the two devices; nullptr if that is not possible
ggml_cuda_p2p_ar * ggml_cuda_p2p_ar_init(const int * devices, size_t n_devices);

void ggml_cuda_p2p_ar_free(ggml_cuda_p2p_ar * ar);

// largest reduction in bytes (F32) the kernel takes
size_t ggml_cuda_p2p_ar_max_bytes();

// in-place sum of tensors[0] and tensors[1], contiguous F32 of the same size on the devices of backends[0] and [1]
// returns false without enqueuing anything if the tensors are not supported
bool ggml_cuda_p2p_ar_allreduce(ggml_cuda_p2p_ar * ar, ggml_backend_t * backends, ggml_tensor ** tensors);
