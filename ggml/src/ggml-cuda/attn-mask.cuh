#pragma once

#include "common.cuh"

void ggml_cuda_op_attn_mask_ranges(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
