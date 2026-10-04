#include "argsort.cuh"
#include "top-k.cuh"

#ifdef GGML_CUDA_USE_CUB
#    ifdef GGML_CUDA_CUB_IS_HIPCUB
#        include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#    else
#        include <cub/cub.cuh>
#        if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#            define CUB_TOP_K_AVAILABLE
#            include <cuda/iterator>
using namespace cub;
#        endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#    endif  // GGML_CUDA_CUB_IS_HIPCUB
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE

#if defined(GGML_USE_HIP)

static __device__ __forceinline__ uint32_t top_k_float_to_ordered(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

template<int BLOCK_SIZE>
static __device__ void top_k_gather_equal(
        const float * src, int * dst, int ncols, uint32_t threshold, int limit, int offset) {
    const int tid = threadIdx.x;
    const int lane = tid % warpSize;
    const int warp = tid / warpSize;
    const int nwarps = BLOCK_SIZE / warpSize;
    __shared__ int warp_counts[32];
    int count = 0;

    for (int base = 0; base < ncols && count < limit; base += BLOCK_SIZE) {
        const int col = base + tid;
        const bool equal = col < ncols && top_k_float_to_ordered(src[col]) == threshold;
        const unsigned long long mask = __ballot(equal);
        if (lane == 0) {
            warp_counts[warp] = __popcll(mask);
        }
        __syncthreads();
        int before = count;
        for (int w = 0; w < nwarps; ++w) {
            if (w < warp) {
                before += warp_counts[w];
            }
            count += warp_counts[w];
        }
        const unsigned long long lane_mask = (1ULL << lane) - 1;
        const int pos = before + __popcll(mask & lane_mask);
        if (equal && pos < limit) {
            dst[offset + pos] = col;
        }
        __syncthreads();
    }
}

struct top_k_radix_state {
    uint32_t prefix;
    uint32_t prefix_mask;
    int rank;
    int greater_count;
    int equal_count;
};

static __global__ void top_k_radix_init(top_k_radix_state * states, int nrows, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row] = {0, 0, k, 0, 0};
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_histogram(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int ncols,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    __shared__ int histogram[NBINS];

    histogram[tid] = 0;
    __syncthreads();

    const top_k_radix_state state = states[row];
    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if ((key & state.prefix_mask) == state.prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    block_histograms[histogram_offset + tid] = histogram[tid];
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_select(
        const int * __restrict__ block_histograms,
        top_k_radix_state * __restrict__ states,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int histogram[NBINS];

    int count = 0;
    for (int row_block = 0; row_block < blocks_per_row; ++row_block) {
        const size_t offset = ((size_t) row * blocks_per_row + row_block) * NBINS;
        count += block_histograms[offset + tid];
    }
    histogram[tid] = count;
    __syncthreads();

    if (tid == 0) {
        top_k_radix_state state = states[row];
        int bin = NBINS - 1;
        while (bin > 0 && histogram[bin] < state.rank) {
            state.rank -= histogram[bin--];
        }
        state.prefix |= (uint32_t) bin << shift;
        state.prefix_mask |= (uint32_t) (NBINS - 1) << shift;
        states[row] = state;
    }
}

static __global__ void top_k_radix_reset_counters(top_k_radix_state * states, int nrows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row].greater_count = 0;
        states[row].equal_count = 0;
    }
}

template<int BLOCK_SIZE, bool STABLE_TIES = false>
static __global__ void top_k_radix_gather(
        const float * __restrict__ src,
        int * __restrict__ dst,
        top_k_radix_state * __restrict__ states,
        int ncols,
        int k,
        int blocks_per_row) {
    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    top_k_radix_state * state = &states[row];

    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if (key > state->prefix) {
            const int pos = atomicAdd(&state->greater_count, 1);
            row_dst[pos] = col;
        } else if constexpr (!STABLE_TIES) {
            if (key == state->prefix) {
                const int pos = atomicAdd(&state->equal_count, 1);
                if (pos < state->rank) {
                    row_dst[k - state->rank + pos] = col;
                }
            }
        }
    }
}

template<int BLOCK_SIZE>
static __global__ void top_k_radix_gather_equal(
        const float * src, int * dst, const top_k_radix_state * states, int ncols, int k) {
    const int row = blockIdx.x;
    const top_k_radix_state state = states[row];
    top_k_gather_equal<BLOCK_SIZE>(src + (size_t) row*ncols, dst + (size_t) row*k,
                                  ncols, state.prefix, state.rank, k - state.rank);
}

static void top_k_radix_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream, bool stable_ties = false) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    const int blocks_per_row = std::min((ncols + 1023) / 1024, 64);

    ggml_cuda_pool_alloc<top_k_radix_state> states_alloc(pool, nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) nrows * blocks_per_row * NBINS);
    top_k_radix_state * states = states_alloc.get();
    int * histograms = histograms_alloc.get();

    top_k_radix_init<<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows, k);

    const dim3 row_grid(blocks_per_row * nrows);
    for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
        top_k_radix_histogram<BLOCK_SIZE, RADIX_BITS>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, states, histograms, ncols, blocks_per_row, shift);
        top_k_radix_select<BLOCK_SIZE, RADIX_BITS>
            <<<nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
    }

    top_k_radix_reset_counters
        <<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows);
    if (stable_ties) {
        top_k_radix_gather<BLOCK_SIZE, true>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(src, dst, states, ncols, k, blocks_per_row);
        top_k_radix_gather_equal<BLOCK_SIZE>
            <<<nrows, BLOCK_SIZE, 0, stream>>>(src, dst, states, ncols, k);
    } else {
        top_k_radix_gather<BLOCK_SIZE>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(src, dst, states, ncols, k, blocks_per_row);
    }
}

// One workgroup per row (RDNA3.5 prefill, e.g. the qwen4exp QSA block selection: 512 of ~62k blocks per query).
// The radix path above reads every row ~5.5 times (four 8-bit histogram passes, a gather and a tie gather); here a
// 12-bit LDS histogram locates the threshold bucket, a second pass writes the keys above it and compacts the bucket
// into LDS, and the remaining 20 bits are resolved on the compacted candidates. A bucket larger than the LDS buffer
// falls back to further histogram passes over the row. The selected set is the same as top_k_radix_cuda with
// stable_ties: every key above the k-th largest, then the lowest-column keys equal to it. Like that path, the order
// of the keys above the threshold is unspecified; the equal keys follow them in column order.
#define TOPK_WG_THREADS 1024
#define TOPK_WG_CAP     4096

// bin b of hist (NBINS bins, descending search) with sum(hist[> b]) < rank <= sum(hist[>= b]); returns b and the
// rank left inside it. Every thread gets the result.
template<int NBINS>
static __device__ __forceinline__ void top_k_wg_find(const int * hist, int rank, int * red, int & bin, int & rank_in) {
    constexpr int PER = NBINS / TOPK_WG_THREADS;
    const int tid = threadIdx.x;
    // thread t owns bins [NBINS - (t+1)*PER, NBINS - t*PER): thread 0 the highest
    int own = 0;
#pragma unroll
    for (int i = 0; i < PER; ++i) { own += hist[NBINS - 1 - (tid * PER + i)]; }
    red[tid] = own;
    __syncthreads();
    // inclusive scan over threads (Hillis-Steele)
    for (int off = 1; off < TOPK_WG_THREADS; off <<= 1) {
        const int v = tid >= off ? red[tid - off] : 0;
        __syncthreads();
        red[tid] += v;
        __syncthreads();
    }
    const int incl = red[tid], excl = incl - own;
    __shared__ int s_bin, s_rank;
    if (excl < rank && rank <= incl) {
        int r = rank - excl, b = NBINS - 1 - tid * PER;
        for (int i = 0; i < PER; ++i, --b) {
            const int h = hist[b];
            if (r <= h) break;
            r -= h;
        }
        s_bin = b; s_rank = r;
    }
    __syncthreads();
    bin = s_bin; rank_in = s_rank;
    __syncthreads();
}

static __global__ void __launch_bounds__(TOPK_WG_THREADS) top_k_wg_kernel(
        const float * __restrict__ src, int * __restrict__ dst, const int ncols, const int k) {
    const int row = blockIdx.x, tid = threadIdx.x;
    const float * x = src + (size_t) row * ncols;
    int * out = dst + (size_t) row * k;

    __shared__ int hist[4096];
    __shared__ int red[TOPK_WG_THREADS];
    __shared__ uint32_t ckey[TOPK_WG_CAP];
    __shared__ int      ccol[TOPK_WG_CAP];
    __shared__ int n_out, n_cand;

    for (int i = tid; i < 4096; i += TOPK_WG_THREADS) hist[i] = 0;
    if (tid == 0) { n_out = 0; n_cand = 0; }
    __syncthreads();

    // pass 1: top 12 bits (four independent loads in flight per thread)
    {
        int c = tid;
        for (; c + 3 * TOPK_WG_THREADS < ncols; c += 4 * TOPK_WG_THREADS) {
            const float v0 = x[c], v1 = x[c + TOPK_WG_THREADS], v2 = x[c + 2 * TOPK_WG_THREADS], v3 = x[c + 3 * TOPK_WG_THREADS];
            atomicAdd(&hist[top_k_float_to_ordered(v0) >> 20], 1);
            atomicAdd(&hist[top_k_float_to_ordered(v1) >> 20], 1);
            atomicAdd(&hist[top_k_float_to_ordered(v2) >> 20], 1);
            atomicAdd(&hist[top_k_float_to_ordered(v3) >> 20], 1);
        }
        for (; c < ncols; c += TOPK_WG_THREADS) {
            atomicAdd(&hist[top_k_float_to_ordered(x[c]) >> 20], 1);
        }
    }
    __syncthreads();
    int b12, rank;
    top_k_wg_find<4096>(hist, k, red, b12, rank);
    const int bucket = hist[b12];
    uint32_t prefix = (uint32_t) b12 << 20, pmask = 0xFFF00000u;
    __syncthreads();

    // pass 2: keys above the bucket go out, the bucket is compacted when it fits
    const bool fits = bucket <= TOPK_WG_CAP;
    auto place = [&](const int c, const float v) {
        const uint32_t key = top_k_float_to_ordered(v);
        const uint32_t top = key >> 20;
        if (top > (uint32_t) b12) {
            out[atomicAdd(&n_out, 1)] = c;
        } else if (fits && top == (uint32_t) b12) {
            const int p = atomicAdd(&n_cand, 1);
            ckey[p] = key; ccol[p] = c;
        }
    };
    {
        int c = tid;
        for (; c + 3 * TOPK_WG_THREADS < ncols; c += 4 * TOPK_WG_THREADS) {
            const float v0 = x[c], v1 = x[c + TOPK_WG_THREADS], v2 = x[c + 2 * TOPK_WG_THREADS], v3 = x[c + 3 * TOPK_WG_THREADS];
            place(c, v0); place(c + TOPK_WG_THREADS, v1); place(c + 2 * TOPK_WG_THREADS, v2); place(c + 3 * TOPK_WG_THREADS, v3);
        }
        for (; c < ncols; c += TOPK_WG_THREADS) place(c, x[c]);
    }
    __syncthreads();

    // remaining 20 bits: 10 + 10, on the candidates or on the row
    const int ncand = fits ? n_cand : 0;
    for (int shift = 10; shift >= 0; shift -= 10) {
        for (int i = tid; i < 1024; i += TOPK_WG_THREADS) hist[i] = 0;
        __syncthreads();
        if (fits) {
            for (int i = tid; i < ncand; i += TOPK_WG_THREADS) {
                const uint32_t key = ckey[i];
                if ((key & pmask) == prefix) atomicAdd(&hist[(key >> shift) & 1023], 1);
            }
        } else {
            for (int c = tid; c < ncols; c += TOPK_WG_THREADS) {
                const uint32_t key = top_k_float_to_ordered(x[c]);
                if ((key & pmask) == prefix) atomicAdd(&hist[(key >> shift) & 1023], 1);
            }
        }
        __syncthreads();
        int b10, r2;
        top_k_wg_find<1024>(hist, rank, red, b10, r2);
        rank = r2;
        prefix |= (uint32_t) b10 << shift;
        pmask  |= 1023u << shift;
        __syncthreads();
    }
    const uint32_t thr = prefix;   // the k-th largest key; `rank` keys equal to it are taken

    // keys above the threshold inside the bucket
    if (fits) {
        for (int i = tid; i < ncand; i += TOPK_WG_THREADS) {
            if (ckey[i] > thr) out[atomicAdd(&n_out, 1)] = ccol[i];
        }
    } else {
        for (int c = tid; c < ncols; c += TOPK_WG_THREADS) {
            const uint32_t key = top_k_float_to_ordered(x[c]);
            if ((key >> 20) == (uint32_t) b12 && key > thr) out[atomicAdd(&n_out, 1)] = c;
        }
    }
    __syncthreads();
    const int base = n_out;   // == k - rank

    // equal keys, lowest columns first
    if (fits) {
        // gather the equal columns, then pick the `rank` smallest by counting (each is unique)
        __shared__ int n_eq;
        if (tid == 0) n_eq = 0;
        __syncthreads();
        // the histogram is free now and holds up to TOPK_WG_CAP (>= ncand) columns
        for (int i = tid; i < ncand; i += TOPK_WG_THREADS) {
            if (ckey[i] == thr) { const int p = atomicAdd(&n_eq, 1); hist[p] = ccol[i]; }
        }
        __syncthreads();
        const int ne = n_eq;
        for (int i = tid; i < ne; i += TOPK_WG_THREADS) {
            const int col = hist[i];
            int less = 0;
            for (int j = 0; j < ne; ++j) less += hist[j] < col;
            if (less < rank) out[base + less] = col;
        }
    } else {
        const int lane = tid % warpSize, warp = tid / warpSize;
        constexpr int NW = TOPK_WG_THREADS / 32;
        int taken = 0;
        for (int c0 = 0; c0 < ncols && taken < rank; c0 += TOPK_WG_THREADS) {
            const int c = c0 + tid;
            const bool eq = c < ncols && top_k_float_to_ordered(x[c]) == thr;
            const unsigned long long m = __ballot(eq);
            if (lane == 0) red[warp] = __popcll(m);
            __syncthreads();
            int before = taken;
            for (int w = 0; w < NW; ++w) { if (w < warp) before += red[w]; }
            int total = 0;
            for (int w = 0; w < NW; ++w) total += red[w];
            const int pos = before + __popcll(m & ((1ULL << lane) - 1));
            if (eq && pos < rank) out[base + pos] = c;
            taken += total;
            __syncthreads();
        }
    }
}

#endif // defined(GGML_USE_HIP)

// RDNA3.5 (results/2026-10-03-overnight/d): two-pass register top-k for small k (<= 32).
// The MTP draft step's backend top-k 10 runs over the scattered full-vocab row (248320 floats); on HIP the default path
// is a hipCUB device radix sort of the whole row plus a copy (~0.22 ms per call, verify-profile item 16). Here pass 1
// splits each row over up to 128 blocks, every thread keeps a sorted register list of its best k keys, and the block
// merges them with k rounds of a block-wide max; pass 2 merges the per-block lists of each row the same way. Keys are
// (order-preserving float bits << 32) | ~col, so ties go to the lower column, as in the stable descending radix sort;
// the output is sorted descending like the argsort fallback it replaces.
static __device__ __forceinline__ uint64_t top_k_fast_key(const float v, const int col) {
    const uint32_t bits = __float_as_uint(v);
    const uint32_t ord  = bits ^ ((uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U);
    return ((uint64_t) ord << 32) | (uint64_t) (0xFFFFFFFFu - (uint32_t) col);
}

template <int KP>
static __device__ __forceinline__ void top_k_fast_insert(uint64_t (&v)[KP], const uint64_t key) {
    if (key <= v[KP - 1]) {
        return;
    }
    v[KP - 1] = key;
#pragma unroll
    for (int j = KP - 1; j > 0; --j) {
        if (v[j] > v[j - 1]) {
            const uint64_t t = v[j]; v[j] = v[j - 1]; v[j - 1] = t;
        }
    }
}

static __device__ __forceinline__ uint64_t top_k_fast_warp_max(uint64_t x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const uint32_t lo = __shfl_xor_sync(0xffffffff, (uint32_t) x,         offset, 32);
        const uint32_t hi = __shfl_xor_sync(0xffffffff, (uint32_t) (x >> 32), offset, 32);
        const uint64_t y  = ((uint64_t) hi << 32) | lo;
        x = y > x ? y : x;
    }
    return x;
}

// block-wide k-way selection over the threads' sorted lists; thread 0 sees each winner in order
template <int KP, int NT, typename F>
static __device__ __forceinline__ void top_k_fast_block_select(uint64_t (&v)[KP], const int k, F && emit) {
    __shared__ uint64_t s_max[NT / 32];
    const int tid = threadIdx.x;
    for (int r = 0; r < k; ++r) {
        const uint64_t m = top_k_fast_warp_max(v[0]);
        if ((tid & 31) == 0) {
            s_max[tid / 32] = m;
        }
        __syncthreads();
        uint64_t best = s_max[0];
#pragma unroll
        for (int w = 1; w < NT / 32; ++w) {
            best = s_max[w] > best ? s_max[w] : best;
        }
        __syncthreads();
        if (v[0] == best && best != 0) {   // keys are unique (they carry the column), so exactly one thread pops
#pragma unroll
            for (int j = 0; j < KP - 1; ++j) {
                v[j] = v[j + 1];
            }
            v[KP - 1] = 0;
        }
        if (tid == 0) {
            emit(r, best);
        }
    }
}

template <int KP, int NT>
static __global__ void top_k_fast_pass1(const float * __restrict__ src, uint64_t * __restrict__ part,
                                        const int ncols, const int k, const int chunk) {
    const int row = blockIdx.y;
    const int blk = blockIdx.x;
    const float * x = src + (size_t) row * ncols;
    uint64_t v[KP];
#pragma unroll
    for (int j = 0; j < KP; ++j) {
        v[j] = 0;
    }
    const int c0 = blk * chunk;
    const int c1 = min(ncols, c0 + chunk);
    for (int c = c0 + threadIdx.x; c < c1; c += NT) {
        top_k_fast_insert<KP>(v, top_k_fast_key(x[c], c));
    }
    uint64_t * out = part + ((size_t) row * gridDim.x + blk) * k;
    top_k_fast_block_select<KP, NT>(v, k, [&](int r, uint64_t key) { out[r] = key; });
}

template <int KP, int NT>
static __global__ void top_k_fast_pass2(const uint64_t * __restrict__ part, int * __restrict__ dst,
                                        const int nparts, const int k) {
    const int row = blockIdx.x;
    const uint64_t * p = part + (size_t) row * nparts * k;
    uint64_t v[KP];
#pragma unroll
    for (int j = 0; j < KP; ++j) {
        v[j] = 0;
    }
    for (int i = threadIdx.x; i < nparts * k; i += NT) {
        top_k_fast_insert<KP>(v, p[i]);
    }
    int * out = dst + (size_t) row * k;
    top_k_fast_block_select<KP, NT>(v, k, [&](int r, uint64_t key) { out[r] = (int) (0xFFFFFFFFu - (uint32_t) key); });
}

template <int KP>
static void top_k_fast_launch(ggml_cuda_pool & pool, const float * src, int * dst, int ncols, int nrows, int k,
                              cudaStream_t stream) {
    constexpr int NT = 256;
    const int nparts = std::max(1, std::min(128, (ncols + 16*NT - 1) / (16*NT)));
    const int chunk  = (ncols + nparts - 1) / nparts;
    ggml_cuda_pool_alloc<uint64_t> part(pool, (size_t) nrows * nparts * k);
    top_k_fast_pass1<KP, NT><<<dim3(nparts, nrows), NT, 0, stream>>>(src, part.get(), ncols, k, chunk);
    top_k_fast_pass2<KP, NT><<<nrows, NT, 0, stream>>>(part.get(), dst, nparts, k);
}

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();
#if defined(GGML_USE_HIP)
    // gfx1151 measurements (2026-09-14) favor radix by 1.2-5.1x above these cutoffs.
    // Recheck the cutoffs when hipCUB changes; smaller batches favor its sort path.
    const bool use_radix = ncols >= 8192 || (ncols >= 4096 && nrows >= 128) || (ncols >= 2048 && nrows >= 512);
    if (GGML_CUDA_CC_IS_RDNA3_5(ggml_cuda_info().devices[ctx.device].cc) && use_radix &&
            ncols <= INT_MAX && nrows > 1 && nrows <= INT_MAX && k <= INT_MAX) {
        // GGML_CUDA_TOPK_WG=0 keeps the multi-pass radix kernels (same selected set)
        static const bool wg = getenv("GGML_CUDA_TOPK_WG") == nullptr || atoi(getenv("GGML_CUDA_TOPK_WG")) != 0;
        if (wg && k <= ncols && nrows <= INT_MAX) {
            top_k_wg_kernel<<<(unsigned) nrows, TOPK_WG_THREADS, 0, stream>>>(src0_d, dst_d, (int) ncols, (int) k);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream, true);
        return;
    }
#endif
    // two-pass register top-k: k <= 32 only, and rows whose finite keys fit the per-thread lists (>= k columns, < 2^31);
    // the multi-row radix cases above keep their path
    if (GGML_CUDA_CC_IS_RDNA3_5(ggml_cuda_info().devices[ctx.device].cc) &&
            k >= 1 && k <= 32 && ncols >= k && ncols > 64 && ncols < INT32_MAX && nrows <= 65535) {
        if (k <= 8) {
            top_k_fast_launch<8>(pool, src0_d, dst_d, (int) ncols, (int) nrows, (int) k, stream);
        } else if (k <= 16) {
            top_k_fast_launch<16>(pool, src0_d, dst_d, (int) ncols, (int) nrows, (int) k, stream);
        } else {
            top_k_fast_launch<32>(pool, src0_d, dst_d, (int) ncols, (int) nrows, (int) k, stream);
        }
        return;
    }
#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    const bool   use_bitonic    = shared_mem <= max_shared_mem && ncols <= 1024;
    const int    chunk_nrows    = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        if (use_bitonic) {
            argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        } else {
            argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src0_d += ncols * iter_nrows;
        dst_d  += k     * iter_nrows;
    }
#else                             // GGML_CUDA_USE_CUB
#if defined(GGML_USE_HIP)
    if (ncols > 1024) {
        // Select tied scores by increasing column index: QSA has many exact-zero ties at the cut.
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream, true);
    } else {
#endif // defined(GGML_USE_HIP)
        ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
        int *                     tmp_dst = temp_dst_alloc.get();
        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                     cudaMemcpyDeviceToDevice, stream));
#if defined(GGML_USE_HIP)
    }
#endif // defined(GGML_USE_HIP)
#endif
}
