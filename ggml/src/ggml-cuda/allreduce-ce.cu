#include "allreduce-ce.cuh"

#if !defined(GGML_USE_MUSA)

#include "convert.cuh"

#include <algorithm>

// A reduction is processed in pieces of one slot. Two slots alternate, so that the next piece can be converted
// while the previous one is still on its way. A slot is sent in chunks: the peer copies a chunk back as soon as it
// is in host memory, so the slower link sets the pace and the faster one follows a chunk behind.
static constexpr int    GGML_CUDA_CE_AR_N_SLOTS     = 2;
static constexpr size_t GGML_CUDA_CE_AR_SLOT_BYTES  = 4u << 20; // BF16 bytes per slot and buffer
static constexpr size_t GGML_CUDA_CE_AR_CHUNK_BYTES = 512u << 10;
static constexpr int    GGML_CUDA_CE_AR_MAX_CHUNKS  = GGML_CUDA_CE_AR_SLOT_BYTES / GGML_CUDA_CE_AR_CHUNK_BYTES;

static constexpr int64_t GGML_CUDA_CE_AR_SLOT_ELEMS  = GGML_CUDA_CE_AR_SLOT_BYTES  / sizeof(nv_bfloat16);
static constexpr int64_t GGML_CUDA_CE_AR_CHUNK_ELEMS = GGML_CUDA_CE_AR_CHUNK_BYTES / sizeof(nv_bfloat16);

struct ggml_cuda_ce_ar {
    int devices[2] = { -1, -1 };

    // the buffers are set up with the first large reduction, a context that never has one allocates nothing (an
    // MTP draft does: it processes the prompt batches as well); if that fails, every call is left to the caller
    bool allocated = false;
    bool failed    = false;

    cudaStream_t d2h[2] = {}; // copies the own contribution to host memory
    cudaStream_t h2d[2] = {}; // copies the contribution of the peer from host memory

    nv_bfloat16 * host[2] = {}; // pinned, written by device i: [slot][elems]
    nv_bfloat16 * own [2] = {}; // device i: the own contribution in BF16
    nv_bfloat16 * peer[2] = {}; // device i: the contribution of the peer

    // [device][slot]: the conversion into own, the copies of own to host (per chunk and all), the copy of the
    // peer's contribution into peer, and the sum that reads own and peer
    cudaEvent_t ev_conv    [2][GGML_CUDA_CE_AR_N_SLOTS]                            = {};
    cudaEvent_t ev_d2h     [2][GGML_CUDA_CE_AR_N_SLOTS][GGML_CUDA_CE_AR_MAX_CHUNKS] = {};
    cudaEvent_t ev_d2h_done[2][GGML_CUDA_CE_AR_N_SLOTS]                            = {};
    cudaEvent_t ev_h2d     [2][GGML_CUDA_CE_AR_N_SLOTS]                            = {};
    cudaEvent_t ev_add     [2][GGML_CUDA_CE_AR_N_SLOTS]                            = {};

    int slot = 0;
};

// both devices add the same two BF16 values (in swapped order, which IEEE addition does not see), so both get the
// same result
static __global__ void ggml_cuda_ce_ar_add(float * dst, const nv_bfloat16 * a, const nv_bfloat16 * b, const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __bfloat162float(a[i]) + __bfloat162float(b[i]);
    }
}

// frees what is allocated and resets the pointers
static void ggml_cuda_ce_ar_release(ggml_cuda_ce_ar * ar) {
    auto destroy_event = [](cudaEvent_t & ev) {
        if (ev) {
            CUDA_CHECK(cudaEventDestroy(ev));
            ev = nullptr;
        }
    };
    for (int i = 0; i < 2; ++i) {
        if (ar->devices[i] < 0) {
            continue;
        }
        ggml_cuda_set_device(ar->devices[i]);
        for (int s = 0; s < GGML_CUDA_CE_AR_N_SLOTS; ++s) {
            destroy_event(ar->ev_conv[i][s]);
            destroy_event(ar->ev_d2h_done[i][s]);
            destroy_event(ar->ev_h2d[i][s]);
            destroy_event(ar->ev_add[i][s]);
            for (int c = 0; c < GGML_CUDA_CE_AR_MAX_CHUNKS; ++c) {
                destroy_event(ar->ev_d2h[i][s][c]);
            }
        }
        if (ar->d2h[i]) {
            CUDA_CHECK(cudaStreamDestroy(ar->d2h[i]));
            ar->d2h[i] = nullptr;
        }
        if (ar->h2d[i]) {
            CUDA_CHECK(cudaStreamDestroy(ar->h2d[i]));
            ar->h2d[i] = nullptr;
        }
        if (ar->own[i]) {
            CUDA_CHECK(cudaFree(ar->own[i]));
            ar->own[i] = nullptr;
        }
        if (ar->peer[i]) {
            CUDA_CHECK(cudaFree(ar->peer[i]));
            ar->peer[i] = nullptr;
        }
        if (ar->host[i]) {
            CUDA_CHECK(cudaFreeHost(ar->host[i]));
            ar->host[i] = nullptr;
        }
    }
}

void ggml_cuda_ce_ar_free(ggml_cuda_ce_ar * ar) {
    if (ar == nullptr) {
        return;
    }
    ggml_cuda_ce_ar_release(ar);
    delete ar;
}

ggml_cuda_ce_ar * ggml_cuda_ce_ar_init(const int * devices, size_t n_devices) {
    if (n_devices != 2) {
        return nullptr;
    }

    auto * ar = new ggml_cuda_ce_ar;
    ar->devices[0] = devices[0];
    ar->devices[1] = devices[1];
    return ar;
}

static bool ggml_cuda_ce_ar_alloc(ggml_cuda_ce_ar * ar) {
    const size_t bytes = GGML_CUDA_CE_AR_N_SLOTS * GGML_CUDA_CE_AR_SLOT_BYTES;

    bool ok = true;
    for (int i = 0; i < 2 && ok; ++i) {
        ggml_cuda_set_device(ar->devices[i]);

        ok = ok && cudaStreamCreateWithFlags(&ar->d2h[i], cudaStreamNonBlocking) == cudaSuccess;
        ok = ok && cudaStreamCreateWithFlags(&ar->h2d[i], cudaStreamNonBlocking) == cudaSuccess;
        ok = ok && cudaMalloc((void **) &ar->own[i],  bytes) == cudaSuccess;
        ok = ok && cudaMalloc((void **) &ar->peer[i], bytes) == cudaSuccess;
        ok = ok && cudaHostAlloc((void **) &ar->host[i], bytes, cudaHostAllocPortable) == cudaSuccess;

        for (int s = 0; s < GGML_CUDA_CE_AR_N_SLOTS && ok; ++s) {
            for (cudaEvent_t * ev : { &ar->ev_conv[i][s], &ar->ev_d2h_done[i][s], &ar->ev_h2d[i][s], &ar->ev_add[i][s] }) {
                ok = ok && cudaEventCreateWithFlags(ev, cudaEventDisableTiming) == cudaSuccess;
            }
            for (int c = 0; c < GGML_CUDA_CE_AR_MAX_CHUNKS && ok; ++c) {
                ok = ok && cudaEventCreateWithFlags(&ar->ev_d2h[i][s][c], cudaEventDisableTiming) == cudaSuccess;
            }
        }
    }

    if (!ok) {
        (void) cudaGetLastError();
        GGML_LOG_WARN("%s: cannot allocate the copy-engine AllReduce buffers, using the other AllReduce\n", __func__);
        ggml_cuda_ce_ar_release(ar);
        return false;
    }

    GGML_LOG_INFO("%s: copy-engine AllReduce: %d slots of %zu KiB per GPU (%zu MiB VRAM and %zu MiB pinned memory per GPU)\n", __func__,
            GGML_CUDA_CE_AR_N_SLOTS, GGML_CUDA_CE_AR_SLOT_BYTES >> 10, (2*bytes) >> 20, bytes >> 20);

    return true;
}

bool ggml_cuda_ce_ar_allreduce(ggml_cuda_ce_ar * ar, ggml_backend_t * backends, ggml_tensor ** tensors) {
    const int64_t ne = ggml_nelements(tensors[0]);
    for (int i = 0; i < 2; ++i) {
        if (tensors[i]->type != GGML_TYPE_F32 || ggml_nelements(tensors[i]) != ne || !ggml_is_contiguously_allocated(tensors[i])) {
            return false;
        }
    }
    if (ne == 0) {
        return true;
    }

    if (!ar->allocated) {
        if (ar->failed) {
            return false;
        }
        ar->allocated = ggml_cuda_ce_ar_alloc(ar);
        ar->failed    = !ar->allocated;
        if (!ar->allocated) {
            return false;
        }
    }

    ggml_backend_cuda_context * cuda_ctx[2];
    bool compute[2];
    for (int i = 0; i < 2; ++i) {
        cuda_ctx[i] = (ggml_backend_cuda_context *) backends[i]->context;
        GGML_ASSERT(cuda_ctx[i]->device == ar->devices[i]);
        compute[i] = (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) != 0;
    }

    const to_bf16_cuda_t to_bf16 = ggml_get_to_bf16_cuda(GGML_TYPE_F32);

    for (int64_t off = 0; off < ne; off += GGML_CUDA_CE_AR_SLOT_ELEMS) {
        const int64_t n        = std::min<int64_t>(GGML_CUDA_CE_AR_SLOT_ELEMS, ne - off);
        const int     n_chunks = (int) ((n + GGML_CUDA_CE_AR_CHUNK_ELEMS - 1) / GGML_CUDA_CE_AR_CHUNK_ELEMS);

        const int s = ar->slot;
        ar->slot = (ar->slot + 1) % GGML_CUDA_CE_AR_N_SLOTS;

        const int64_t s_off = s*GGML_CUDA_CE_AR_SLOT_ELEMS;

        // the own contribution to host memory
        for (int i = 0; i < 2; ++i) {
            ggml_cuda_set_device(ar->devices[i]);
            cudaStream_t stream = cuda_ctx[i]->stream();

            nv_bfloat16 * own = ar->own[i] + s_off;

            // own is read by the copy to host of the previous use of the slot
            CUDA_CHECK(cudaStreamWaitEvent(stream, ar->ev_d2h_done[i][s]));
            if (compute[i]) {
                to_bf16((const float *) tensors[i]->data + off, own, n, stream);
            } else {
                // a device with an empty slice adds nothing
                CUDA_CHECK(cudaMemsetAsync(own, 0, n*sizeof(nv_bfloat16), stream));
            }
            CUDA_CHECK(cudaEventRecord(ar->ev_conv[i][s], stream));

            // the host buffer is read by the peer's copy of the previous use of the slot
            CUDA_CHECK(cudaStreamWaitEvent(ar->d2h[i], ar->ev_conv[i][s]));
            CUDA_CHECK(cudaStreamWaitEvent(ar->d2h[i], ar->ev_h2d[1 - i][s]));
            for (int c = 0; c < n_chunks; ++c) {
                const int64_t c_off = c*GGML_CUDA_CE_AR_CHUNK_ELEMS;
                const int64_t c_n   = std::min<int64_t>(GGML_CUDA_CE_AR_CHUNK_ELEMS, n - c_off);
                CUDA_CHECK(cudaMemcpyAsync(ar->host[i] + s_off + c_off, own + c_off, c_n*sizeof(nv_bfloat16),
                        cudaMemcpyDeviceToHost, ar->d2h[i]));
                CUDA_CHECK(cudaEventRecord(ar->ev_d2h[i][s][c], ar->d2h[i]));
            }
            CUDA_CHECK(cudaEventRecord(ar->ev_d2h_done[i][s], ar->d2h[i]));
        }

        // the peer's contribution, chunk by chunk as it arrives, then the sum
        for (int i = 0; i < 2; ++i) {
            const int p = 1 - i;
            ggml_cuda_set_device(ar->devices[i]);
            cudaStream_t stream = cuda_ctx[i]->stream();

            nv_bfloat16 * peer = ar->peer[i] + s_off;

            // peer is read by the sum of the previous use of the slot
            CUDA_CHECK(cudaStreamWaitEvent(ar->h2d[i], ar->ev_add[i][s]));
            for (int c = 0; c < n_chunks; ++c) {
                const int64_t c_off = c*GGML_CUDA_CE_AR_CHUNK_ELEMS;
                const int64_t c_n   = std::min<int64_t>(GGML_CUDA_CE_AR_CHUNK_ELEMS, n - c_off);
                CUDA_CHECK(cudaStreamWaitEvent(ar->h2d[i], ar->ev_d2h[p][s][c]));
                CUDA_CHECK(cudaMemcpyAsync(peer + c_off, ar->host[p] + s_off + c_off, c_n*sizeof(nv_bfloat16),
                        cudaMemcpyHostToDevice, ar->h2d[i]));
            }
            CUDA_CHECK(cudaEventRecord(ar->ev_h2d[i][s], ar->h2d[i]));

            CUDA_CHECK(cudaStreamWaitEvent(stream, ar->ev_h2d[i][s]));
            const int block_size = 256;
            const int64_t n_blocks = (n + block_size - 1) / block_size;
            ggml_cuda_ce_ar_add<<<n_blocks, block_size, 0, stream>>>((float *) tensors[i]->data + off, ar->own[i] + s_off, peer, n);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(ar->ev_add[i][s], stream));
        }
    }

    return true;
}

#else // defined(GGML_USE_MUSA)

// MUSA lacks cudaHostAlloc / cudaHostAllocPortable (see allreduce.cu): the caller keeps its other AllReduce
ggml_cuda_ce_ar * ggml_cuda_ce_ar_init(const int *, size_t) {
    return nullptr;
}
void ggml_cuda_ce_ar_free(ggml_cuda_ce_ar *) {
}
bool ggml_cuda_ce_ar_allreduce(ggml_cuda_ce_ar *, ggml_backend_t *, ggml_tensor **) {
    return false;
}

#endif // !defined(GGML_USE_MUSA)
