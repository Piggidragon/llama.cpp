#include "attn-mask.cuh"
#include "convert.cuh"

// one block row per (row, mask sequence), the columns of a row split between the blocks of x
template <typename T>
static __global__ void attn_mask_ranges(
        const int32_t * __restrict__ r, T * __restrict__ dst, const int64_t n_cols,
        const int64_t s1, const int64_t s3, const int64_t sr1, const int64_t sr3) {
    const int64_t i1 = blockIdx.y;
    const int64_t i3 = blockIdx.z;

    const int32_t * range = r + i1*sr1 + i3*sr3;
    const int64_t lo = range[0];
    const int64_t hi = range[1];

    T * row = dst + i1*s1 + i3*s3;
    for (int64_t j = int64_t(blockIdx.x)*blockDim.x + threadIdx.x; j < n_cols; j += int64_t(gridDim.x)*blockDim.x) {
        row[j] = ggml_cuda_cast<T>(j >= lo && j < hi ? 0.0f : -INFINITY);
    }
}

void ggml_cuda_op_attn_mask_ranges(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * r = dst->src[0];
    GGML_ASSERT(r->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->ne[1] <= 65535 && dst->ne[3] <= 65535);

    const int64_t n_cols = dst->ne[0];
    const int64_t sr1 = r->nb[1]/sizeof(int32_t);
    const int64_t sr3 = r->nb[3]/sizeof(int32_t);

    const int block = 256;
    const dim3 blocks(std::min<int64_t>((n_cols + block - 1)/block, 64), dst->ne[1], dst->ne[3]);
    cudaStream_t stream = ctx.stream();

    if (dst->type == GGML_TYPE_F16) {
        attn_mask_ranges<half><<<blocks, block, 0, stream>>>((const int32_t *) r->data, (half *) dst->data, n_cols,
            dst->nb[1]/sizeof(half), dst->nb[3]/sizeof(half), sr1, sr3);
    } else {
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        attn_mask_ranges<float><<<blocks, block, 0, stream>>>((const int32_t *) r->data, (float *) dst->data, n_cols,
            dst->nb[1]/sizeof(float), dst->nb[3]/sizeof(float), sr1, sr3);
    }
    CUDA_CHECK(cudaGetLastError());
}
