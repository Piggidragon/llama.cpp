#include "allreduce-p2p.cuh"

#if !defined(GGML_USE_MUSA)

#include <algorithm>

// Each AllReduce is one kernel per GPU. Block b of GPU i writes its slice of the own contribution (BF16) into the
// receive buffer of the other GPU and then its token into the flag of the other GPU, both through peer access;
// then it waits for its own flag, which the other GPU writes the same way, and adds. Only local memory is polled,
// and the data crosses the link once, as posted writes.
//
// Two slots alternate. GPU i writes slot k%2 of call k only after its kernel of call k-1 saw the arrival of the
// other GPU for call k-1, which that GPU signals after its kernel of call k-2 has read slot k%2 (stream order).
// The tokens grow, so a flag that already holds a later token also counts as arrived. They wrap around at 2^32:
// a flag counts as arrived when it is at most 2^31 - 1 ahead of the token, in unsigned arithmetic.

static constexpr int     GGML_CUDA_P2P_AR_BLOCKS  = 8;
static constexpr int     GGML_CUDA_P2P_AR_THREADS = 256;
static constexpr int64_t GGML_CUDA_P2P_AR_MAX_NE  = 512*1024/sizeof(float); // the small reductions only

struct ggml_cuda_p2p_ar {
    int devices[2] = { -1, -1 };

    nv_bfloat16 * recv [2] = {}; // device i: [slot][GGML_CUDA_P2P_AR_MAX_NE], written by the other GPU
    unsigned int * flags[2] = {}; // device i: [block], written by the other GPU

    unsigned int token = 0;
};

static __global__ void ggml_cuda_p2p_ar_kernel(
        float * data, const bool compute, nv_bfloat16 * peer_recv, unsigned int * peer_flags,
        const nv_bfloat16 * own_recv, const unsigned int * own_flags, const int64_t n, const unsigned int token) {
    const int64_t per   = (n + gridDim.x - 1) / gridDim.x;
    const int64_t i0    = blockIdx.x*per;
    const int64_t i1    = i0 + per < n ? i0 + per : n;

    for (int64_t i = i0 + threadIdx.x; i < i1; i += blockDim.x) {
        peer_recv[i] = compute ? __float2bfloat16(data[i]) : __float2bfloat16(0.0f);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        // the data before the token, for the other GPU
        __threadfence_system();
        *(volatile unsigned int *) &peer_flags[blockIdx.x] = token;

        while (*(const volatile unsigned int *) &own_flags[blockIdx.x] - token > 0x7fffffffu) {
        }
        __threadfence();
    }
    __syncthreads();

    for (int64_t i = i0 + threadIdx.x; i < i1; i += blockDim.x) {
        const float own = compute ? __bfloat162float(__float2bfloat16(data[i])) : 0.0f;
        data[i] = own + __bfloat162float(own_recv[i]);
    }
}

void ggml_cuda_p2p_ar_free(ggml_cuda_p2p_ar * ar) {
    if (ar == nullptr) {
        return;
    }
    for (int i = 0; i < 2; ++i) {
        if (ar->devices[i] < 0) {
            continue;
        }
        ggml_cuda_set_device(ar->devices[i]);
        if (ar->recv[i]) {
            CUDA_CHECK(cudaFree(ar->recv[i]));
        }
        if (ar->flags[i]) {
            CUDA_CHECK(cudaFree(ar->flags[i]));
        }
    }
    delete ar;
}

size_t ggml_cuda_p2p_ar_max_bytes() {
    return GGML_CUDA_P2P_AR_MAX_NE*sizeof(float);
}

ggml_cuda_p2p_ar * ggml_cuda_p2p_ar_init(const int * devices, size_t n_devices) {
    if (n_devices != 2) {
        return nullptr;
    }

    auto * ar = new ggml_cuda_p2p_ar;
    ar->devices[0] = devices[0];
    ar->devices[1] = devices[1];

    bool ok = true;
    for (int i = 0; i < 2 && ok; ++i) {
        int can_access = 0;
        ok = cudaDeviceCanAccessPeer(&can_access, devices[i], devices[1 - i]) == cudaSuccess && can_access;
        if (!ok) {
            break;
        }
        ggml_cuda_set_device(devices[i]);
        const cudaError_t err = cudaDeviceEnablePeerAccess(devices[1 - i], 0);
        if (err == cudaErrorPeerAccessAlreadyEnabled) {
            (void) cudaGetLastError();
        } else {
            ok = err == cudaSuccess;
        }
        ok = ok && cudaMalloc((void **) &ar->recv[i], 2*GGML_CUDA_P2P_AR_MAX_NE*sizeof(nv_bfloat16)) == cudaSuccess;
        ok = ok && cudaMalloc((void **) &ar->flags[i], GGML_CUDA_P2P_AR_BLOCKS*sizeof(unsigned int)) == cudaSuccess;
        ok = ok && cudaMemset(ar->flags[i], 0, GGML_CUDA_P2P_AR_BLOCKS*sizeof(unsigned int)) == cudaSuccess;
    }

    if (!ok) {
        (void) cudaGetLastError();
        GGML_LOG_WARN("%s: cannot set up the peer-to-peer AllReduce\n", __func__);
        ggml_cuda_p2p_ar_free(ar);
        return nullptr;
    }

    GGML_LOG_INFO("%s: peer-to-peer AllReduce up to %zu bytes (%zu KiB VRAM per GPU)\n", __func__,
            ggml_cuda_p2p_ar_max_bytes(), (2*GGML_CUDA_P2P_AR_MAX_NE*sizeof(nv_bfloat16)) >> 10);
    return ar;
}

bool ggml_cuda_p2p_ar_allreduce(ggml_cuda_p2p_ar * ar, ggml_backend_t * backends, ggml_tensor ** tensors) {
    const int64_t ne = ggml_nelements(tensors[0]);
    for (int i = 0; i < 2; ++i) {
        if (tensors[i]->type != GGML_TYPE_F32 || ggml_nelements(tensors[i]) != ne || !ggml_is_contiguously_allocated(tensors[i])) {
            return false;
        }
    }
    if (ne > GGML_CUDA_P2P_AR_MAX_NE) {
        return false;
    }
    if (ne == 0) {
        return true;
    }

    ar->token++;
    const unsigned int token = ar->token;
    const int64_t s_off = (ar->token % 2)*GGML_CUDA_P2P_AR_MAX_NE;

    for (int i = 0; i < 2; ++i) {
        const int p = 1 - i;
        auto * cuda_ctx = (ggml_backend_cuda_context *) backends[i]->context;
        GGML_ASSERT(cuda_ctx->device == ar->devices[i]);
        ggml_cuda_set_device(ar->devices[i]);

        const bool compute = (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) != 0;
        ggml_cuda_p2p_ar_kernel<<<GGML_CUDA_P2P_AR_BLOCKS, GGML_CUDA_P2P_AR_THREADS, 0, cuda_ctx->stream()>>>(
                (float *) tensors[i]->data, compute, ar->recv[p] + s_off, ar->flags[p], ar->recv[i] + s_off, ar->flags[i], ne, token);
        CUDA_CHECK(cudaGetLastError());
    }

    return true;
}

#else // defined(GGML_USE_MUSA)

ggml_cuda_p2p_ar * ggml_cuda_p2p_ar_init(const int *, size_t) {
    return nullptr;
}
void ggml_cuda_p2p_ar_free(ggml_cuda_p2p_ar *) {
}
size_t ggml_cuda_p2p_ar_max_bytes() {
    return 0;
}
bool ggml_cuda_p2p_ar_allreduce(ggml_cuda_p2p_ar *, ggml_backend_t *, ggml_tensor **) {
    return false;
}

#endif // !defined(GGML_USE_MUSA)
