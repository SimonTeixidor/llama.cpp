// W4A16 prefill GEMM for RDNA3.5 (gfx1151).
//
// Adapted from gufo (github.com/gufo-org/gufo, MIT licence), commit b722a61,
// src/models/qwen/hip/kernels/prefill_fp16.hip (HalfPrefillGemmKernel, LoadSwizzled/StoreSwizzled,
// DecodeIqRaw). The tiling, LDS layout + XOR swizzle, block rasterisation and the IQ4 codebook
// byte-permute decode follow that file; the ggml integration, type plumbing and tails are ours.
//
// dst[t][r] = sum_k W[r][k] * X[t][k] with W decoded to fully scaled FP16 while staging into LDS,
// X converted once from F32 to FP16, FP32 accumulation (no per-block rescale of the accumulators).

#include "mmw4a16.cuh"
#include "mmq.cuh"
#include "convert.cuh"

#if defined(GGML_USE_HIP) && defined(RDNA3)
#define W4A16_DEVICE
#endif

typedef _Float16 w4a16_h16 __attribute__((ext_vector_type(16)));
typedef float    w4a16_f8  __attribute__((ext_vector_type(8)));

// Permute the two 16-byte halves of a 32-byte LDS row to avoid repeated bank conflicts (gufo).
static __device__ __forceinline__ w4a16_h16 w4a16_load_swz(const short * p) {
    union { w4a16_h16 f; uint4 q[2]; } bits;
    const unsigned row = threadIdx.x & 15, shift = (row ^ (row >> 2)) & 1;
    bits.q[0] = reinterpret_cast<const uint4 *>(p)[shift];
    bits.q[1] = reinterpret_cast<const uint4 *>(p)[shift ^ 1];
    return bits.f;
}

static __device__ __forceinline__ void w4a16_store_swz(short * p, uint4 lo, uint4 hi, unsigned row) {
    const unsigned shift = (row ^ (row >> 2)) & 1;
    reinterpret_cast<uint4 *>(p)[shift]     = lo;
    reinterpret_cast<uint4 *>(p)[shift ^ 1] = hi;
}

// Raw (still packed) 16 weights of one IQ4_XS K16 sub-block + their fp32 scale.
struct w4a16_iq_raw {
    uint32_t data[4];
    uint32_t shift;
    float    scale;
};

// sub = K16 index within the row. Header (d | scales_h<<16, scales_l) is cached by the caller per 256-block.
static __device__ __forceinline__ w4a16_iq_raw w4a16_load_iq4_xs(const uint8_t * row, unsigned sub, const uint2 & hdr) {
    w4a16_iq_raw r;
    r.shift = (sub & 1) * 4;
    const block_iq4_xs * b = reinterpret_cast<const block_iq4_xs *>(row) + sub / 16;
    const unsigned sb = (sub / 2) & 7;
    __builtin_memcpy(r.data, b->qs + sb * 16, 16);
    const unsigned low  = hdr.y >> (8 * (sb / 2));
    const unsigned high = hdr.x >> 16;
    const int code = (int) (((low >> (4 * (sb & 1))) & 15) | (((high >> (2 * sb)) & 3) << 4)) - 32;
    r.scale = __half2float(__ushort_as_half((unsigned short) (hdr.x & 0xFFFF))) * (float) code;
    return r;
}

// All 16 IQ4 codebook entries are integers, exact in FP16: look their half bits up with v_perm (gufo).
// Result = fp16(scale * codebook[q]), same rounding as ggml's dequantize_row_iq4_xs -> fp16.
static __device__ __forceinline__ void w4a16_decode_iq(const w4a16_iq_raw & r, uint4 & out_lo, uint4 & out_hi) {
    union { w4a16_h16 f; uint4 u[2]; } res;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned n      = (r.data[i] >> r.shift) & 0x0f0f0f0fU;
        const unsigned index  = n & 0x07070707U;
        const unsigned select = ((n & 0x08080808U) >> 3) * 255;
        // low / high bytes of fp16 {-127,-104,-83,-65,-49,-35,-22,-10, 1,13,25,38,53,69,89,113}
        const unsigned lo0 = __builtin_amdgcn_perm(0x00806020U, 0x103080f0U, index);
        const unsigned lo1 = __builtin_amdgcn_perm(0x109050a0U, 0xc0408000U, index);
        const unsigned hi0 = __builtin_amdgcn_perm(0xc9cdd0d2U, 0xd4d5d6d7U, index);
        const unsigned hi1 = __builtin_amdgcn_perm(0x57555452U, 0x504e4a3cU, index);
        const unsigned lo  = (lo0 & ~select) | (lo1 & select);
        const unsigned hi  = (hi0 & ~select) | (hi1 & select);
        const unsigned c0  = __builtin_amdgcn_perm(hi, lo, 0x05010400U);
        const unsigned c1  = __builtin_amdgcn_perm(hi, lo, 0x07030602U);
        res.f[4*i + 0] = (_Float16) __fmul_rn(r.scale, __half2float(__ushort_as_half((unsigned short) (c0 & 0xFFFF))));
        res.f[4*i + 1] = (_Float16) __fmul_rn(r.scale, __half2float(__ushort_as_half((unsigned short) (c0 >> 16))));
        res.f[4*i + 2] = (_Float16) __fmul_rn(r.scale, __half2float(__ushort_as_half((unsigned short) (c1 & 0xFFFF))));
        res.f[4*i + 3] = (_Float16) __fmul_rn(r.scale, __half2float(__ushort_as_half((unsigned short) (c1 >> 16))));
    }
    out_lo = res.u[0];
    out_hi = res.u[1];
}

// Other weight types: raw words are fetched one stage ahead, bit extraction + scaling happen at LDS commit
// (same split as gufo's LoadRaw/DecodeRaw), so the fetch never waits on its own global loads.
struct w4a16_raw {
    uint32_t lo[4];
    uint32_t hi[4];
    uint32_t h0, h1, h2, h3;
    uint32_t sub;
};

template <ggml_type type>
static __device__ __forceinline__ void w4a16_load_raw(const uint8_t * row, const unsigned sub, w4a16_raw & r) {
    r.sub = sub;
    if constexpr (type == GGML_TYPE_Q8_0) {
        const uint8_t * b = row + (sub/2)*sizeof(block_q8_0);
        r.h0 = *reinterpret_cast<const uint16_t *>(b);
        __builtin_memcpy(r.lo, b + 2 + (sub & 1)*16, 16);
    } else if constexpr (type == GGML_TYPE_Q4_0) {
        const uint8_t * b = row + (sub/2)*sizeof(block_q4_0);
        r.h0 = *reinterpret_cast<const uint16_t *>(b);
        __builtin_memcpy(r.lo, b + 2, 16);
    } else if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
        constexpr int bs = type == GGML_TYPE_Q4_K ? sizeof(block_q4_K) : sizeof(block_q5_K);
        constexpr int qs = type == GGML_TYPE_Q4_K ? 16 : 16 + 32;
        const uint8_t * b = row + (sub/16)*bs;
        const unsigned sb = (sub/2) & 7;
        __builtin_memcpy(&r.h0, b, 4);                 // d | dmin << 16
        r.h1 = b[4 + (sb & 3)];                        // scales[sb&3]
        r.h2 = b[4 + (sb & 3) + 4];                    // scales[(sb&3)+4]
        r.h3 = b[4 + (sb & 3) + 8];                    // scales[(sb&3)+8]
        __builtin_memcpy(r.lo, b + qs + (sb/2)*32 + (sub & 1)*16, 16);
        if constexpr (type == GGML_TYPE_Q5_K) {
            __builtin_memcpy(r.hi, b + 16 + (sub & 1)*16, 16);
        }
    } else if constexpr (type == GGML_TYPE_Q6_K) {
        const uint8_t * b = row + (sub/16)*sizeof(block_q6_K);
        const unsigned sb = (sub/2) & 7, half = sb/4, seg = sb % 4;
        __builtin_memcpy(r.lo, b + half*64 + (seg & 1)*32 + (sub & 1)*16, 16);
        __builtin_memcpy(r.hi, b + 128 + half*32 + (sub & 1)*16, 16);
        r.h1 = b[192 + half*8 + seg*2 + (sub & 1)];     // int8 scale
        r.h0 = *reinterpret_cast<const uint16_t *>(b + 208);
    } else {
        static_assert(type == GGML_TYPE_Q8_0, "unsupported type");
    }
}

static __device__ __forceinline__ float w4a16_h2f(const uint32_t bits) {
    return __half2float(__ushort_as_half((unsigned short) (bits & 0xFFFF)));
}

// 16 weights -> fp16(scale*q - offset), matching ggml's dequantize_row_* -> fp16 rounding.
template <ggml_type type>
static __device__ __forceinline__ void w4a16_decode_raw(const w4a16_raw & r, uint4 & out_lo, uint4 & out_hi) {
    float scale, offset = 0.0f;
    int q[16];
    const unsigned sub = r.sub;
    if constexpr (type == GGML_TYPE_Q8_0) {
        scale = w4a16_h2f(r.h0);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            q[i] = (int8_t) ((r.lo[i/4] >> (8*(i % 4))) & 0xFF);
        }
    } else if constexpr (type == GGML_TYPE_Q4_0) {
        scale = w4a16_h2f(r.h0);
        const unsigned shift = (sub & 1)*4;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            q[i] = (int) ((r.lo[i/4] >> (8*(i % 4) + shift)) & 0xF) - 8;
        }
    } else if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
        const unsigned sb = (sub/2) & 7;
        const unsigned sc0 = r.h1, sc1 = r.h2, sc2 = r.h3;
        const unsigned sc = sb < 4 ? (sc0 & 63) : ((sc2 & 15) | ((sc0 >> 6) << 4));
        const unsigned mn = sb < 4 ? (sc1 & 63) : ((sc2 >> 4) | ((sc1 >> 6) << 4));
        scale  = w4a16_h2f(r.h0)       * (float) sc;
        offset = w4a16_h2f(r.h0 >> 16) * (float) mn;
        const unsigned shift = 4*(sb & 1);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            int v = (r.lo[i/4] >> (8*(i % 4) + shift)) & 0xF;
            if constexpr (type == GGML_TYPE_Q5_K) {
                v |= ((r.hi[i/4] >> (8*(i % 4) + sb)) & 1) << 4;
            }
            q[i] = v;
        }
    } else if constexpr (type == GGML_TYPE_Q6_K) {
        const unsigned sb = (sub/2) & 7, seg = sb % 4;
        scale = w4a16_h2f(r.h0) * (float) (int8_t) (r.h1 & 0xFF);
        const unsigned lshift = seg >= 2 ? 4 : 0;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int lo = (r.lo[i/4] >> (8*(i % 4) + lshift)) & 0xF;
            const int hi = (r.hi[i/4] >> (8*(i % 4) + 2*seg)) & 0x3;
            q[i] = (lo | (hi << 4)) - 32;
        }
    }
    union { w4a16_h16 f; uint4 u[2]; } res;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        res.f[i] = (_Float16) (scale*(float) q[i] - offset);
    }
    out_lo = res.u[0];
    out_hi = res.u[1];
}

// BM weight rows x BN tokens per workgroup, WM x WN waves, K64 per LDS stage.
// PAIRED: w = gate, w_up = up; physical tile rows alternate 16 gate / 16 up rows of the same logical rows,
// so one wave owns both accumulators and the epilogue writes silu(gate)*up directly (gufo kGateUp).
template <int BM, int BN, int WM, int WN, ggml_type type, int GROUP_SHIFT, bool PAIRED>
__launch_bounds__(WM*WN*32) __global__ void mul_mat_w4a16(
        const char * __restrict__ w, const char * __restrict__ w_up, const short * __restrict__ x, float * __restrict__ y,
        const int n, const int m, const int k, const int64_t w_row_bytes, const int64_t stride_y) {
#ifdef W4A16_DEVICE
    constexpr bool is_iq4 = type == GGML_TYPE_IQ4_XS;
    // One K64 LDS stage per iteration, gufo's schedule: commit (decode + store) -> barrier -> WMMA -> barrier.
    // Measured on gfx1151 (results/2026-09-28-w4a16-prefill/): a double-buffered K32 variant, smaller co-resident
    // workgroups, 128x512 tiles and a cheaper packed-fp16 IQ4 decode were all 2-15 % slower than this.
    constexpr int BK = 4, NBUF = 1, KS = 16, WS = 32;
    constexpr int kThreads = WM*WN*WS;
    constexpr int RS = BM/16, TS = BN/16, WRS = RS/WM, WTS = TS/WN;
    static_assert(BM % (16*WM) == 0 && BN % (16*WN) == 0, "bad tile");
    constexpr int PFA = (BM*BK + kThreads - 1)/kThreads;
    constexpr int PFB = (BN*BK + kThreads - 1)/kThreads;

    __shared__ short s_a[NBUF][BK][RS][16][KS];
    __shared__ short s_b[NBUF][BK][TS][16][KS];

    const int tid  = threadIdx.x;
    const int wave = tid / WS;
    const int lane = tid % WS;
    const int sl   = lane & 15;
    const int wr   = wave / WN;
    const int wt   = wave % WN;

    // grouped rasterisation: GROUP row tiles walk all token tiles together (input reuse in L2)
    unsigned row_tile, token_tile;
    {
        constexpr unsigned group = 1u << GROUP_SHIFT;
        const unsigned first  = (blockIdx.y >> GROUP_SHIFT) << GROUP_SHIFT;
        const unsigned within = (blockIdx.y & (group - 1))*gridDim.x + blockIdx.x;
        const unsigned rows   = min(group, gridDim.y - first);
        row_tile   = first + within % rows;
        token_tile = within / rows;
    }
    const int r_block = row_tile*(PAIRED ? BM/2 : BM);
    const int t_block = token_tile*BN;
    const int ksteps  = k/16;

    w4a16_f8 acc[WRS][WTS];
#pragma unroll
    for (int i = 0; i < WRS; ++i) {
#pragma unroll
        for (int j = 0; j < WTS; ++j) {
            acc[i][j] = w4a16_f8{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        }
    }

    // row bases clamped so out-of-range rows/tokens read valid memory and are masked at the store
    const uint8_t * w_row[PFA];
#pragma unroll
    for (int p = 0; p < PFA; ++p) {
        const int pr = (p*kThreads + tid)/BK;
        const int r  = r_block + (PAIRED ? (pr/32)*16 + pr%16 : pr);
        const char * src = PAIRED && pr % 32 >= 16 ? w_up : w;
        w_row[p] = reinterpret_cast<const uint8_t *>(src) + (int64_t) (r < m ? r : m - 1)*w_row_bytes;
    }
    const short * x_row[PFB];
#pragma unroll
    for (int p = 0; p < PFB; ++p) {
        const int t = t_block + (p*kThreads + tid)/BK;
        x_row[p] = x + (int64_t) (t < n ? t : n - 1)*k;
    }

    uint4        rb[PFB][2];
    uint2        hdr[PFA];
    w4a16_iq_raw raw[PFA];
    w4a16_raw    graw[PFA];

    const auto fetch = [&](const int ks0) {
#pragma unroll
        for (int p = 0; p < PFA; ++p) {
            const int idx = p*kThreads + tid;
            const int kk  = ks0 + idx % BK;
            if (idx < BM*BK) {
                if constexpr (is_iq4) {
                    if ((ks0 & 15) == 0) { // first stage of a 256-block: refresh the cached header
                        __builtin_memcpy(&hdr[p], reinterpret_cast<const block_iq4_xs *>(w_row[p]) + kk/16, 8);
                    }
                    raw[p] = w4a16_load_iq4_xs(w_row[p], kk, hdr[p]);
                } else {
                    w4a16_load_raw<type>(w_row[p], kk, graw[p]);
                }
            }
        }
#pragma unroll
        for (int p = 0; p < PFB; ++p) {
            const int idx = p*kThreads + tid;
            const int kk  = ks0 + idx % BK;
            if (idx < BN*BK) {
                const short * src = x_row[p] + (int64_t) kk*16;
                rb[p][0] = reinterpret_cast<const uint4 *>(src)[0];
                rb[p][1] = reinterpret_cast<const uint4 *>(src)[1];
            }
        }
    };

    const auto commit = [&](const int buf) {
#pragma unroll
        for (int p = 0; p < PFA; ++p) {
            const int idx = p*kThreads + tid, row = idx / BK, ks = idx % BK;
            if (idx < BM*BK) {
                const int placed = row ^ (ks & 3);
                uint4 lo, hi;
                if constexpr (is_iq4) {
                    w4a16_decode_iq(raw[p], lo, hi);
                } else {
                    w4a16_decode_raw<type>(graw[p], lo, hi);
                }
                w4a16_store_swz(&s_a[buf][ks][placed/16][placed%16][0], lo, hi, row);
            }
        }
#pragma unroll
        for (int p = 0; p < PFB; ++p) {
            const int idx = p*kThreads + tid, row = idx / BK, ks = idx % BK;
            if (idx < BN*BK) {
                const int placed = row ^ (ks & 3);
                w4a16_store_swz(&s_b[buf][ks][placed/16][placed%16][0], rb[p][0], rb[p][1], row);
            }
        }
    };

    const auto compute = [&](const int buf) {
#pragma unroll
        for (int ks = 0; ks < BK; ++ks) {
            w4a16_h16 a[WRS];
#pragma unroll
            for (int i = 0; i < WRS; ++i) {
                a[i] = w4a16_load_swz(&s_a[buf][ks][wr*WRS + i][sl ^ (ks & 3)][0]);
            }
#pragma unroll
            for (int j = 0; j < WTS; ++j) {
                const w4a16_h16 b = w4a16_load_swz(&s_b[buf][ks][wt*WTS + j][sl ^ (ks & 3)][0]);
#pragma unroll
                for (int i = 0; i < WRS; ++i) {
                    // A = tokens, B = weight rows -> C[token][row]: lane%16 = row, element l = token 2l + lane/16
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(b, a[i], acc[i][j]);
                }
            }
        }
    };

    fetch(0);
    for (int ks0 = 0; ks0 < ksteps; ks0 += BK) {
        commit(0);
        __syncthreads();
        if (ks0 + BK < ksteps) {
            fetch(ks0 + BK);
        }
        compute(0);
        __builtin_amdgcn_iglp_opt(0);
        __syncthreads();
    }

    const int hi = lane / 16;
    if constexpr (PAIRED) {
        static_assert(WRS % 2 == 0, "paired tile needs an even number of row fragments per wave");
#pragma unroll
        for (int i = 0; i < WRS; i += 2) {
#pragma unroll
            for (int j = 0; j < WTS; ++j) {
                const int row = r_block + (wr*(WRS/2) + i/2)*16 + sl;
                const int t0  = t_block + (wt*WTS + j)*16;
#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    const int token = t0 + 2*l + hi;
                    if (row < m && token < n) {
                        const float g = acc[i][j][l];
                        const float u = acc[i + 1][j][l];
                        y[(int64_t) token*stride_y + row] = g/(1.0f + expf(-g)) * u;
                    }
                }
            }
        }
        return;
    }
#pragma unroll
    for (int i = 0; i < WRS; ++i) {
#pragma unroll
        for (int j = 0; j < WTS; ++j) {
            const int row = r_block + (wr*WRS + i)*16 + sl;
            const int t0  = t_block + (wt*WTS + j)*16;
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const int token = t0 + 2*l + hi;
                if (row < m && token < n) {
                    y[(int64_t) token*stride_y + row] = acc[i][j][l];
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(w, w_up, x, y, n, m, k, w_row_bytes, stride_y);
    NO_DEVICE_CODE;
#endif // W4A16_DEVICE
}

bool ggml_cuda_should_use_w4a16(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc) {
    if (ggml_cuda_w4a16_mode() != 1 || !GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return false;
    }
    switch (src0->type) {
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            break;
        default:
            return false;
    }
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src1->ne[1] < ggml_cuda_w4a16_min_batch()) {
        return false;
    }
    // Weights with fewer than 512 rows leave most of the 256-row tile idle and launch only n/256 workgroups: qwen35moe's
    // quantised ssm_alpha / ssm_beta (m = 32, 60 GEMMs per ubatch) stay on MMQ. GGML_HIP_W4A16_MIN_ROWS=N overrides
    // the threshold (0 = no threshold). The 512-row shared-expert gate/up and attn_k/v stay on W4A16.
    {
        static const int64_t min_rows = [] {
            const char * e = getenv("GGML_HIP_W4A16_MIN_ROWS");
            return e ? (int64_t) atoll(e) : (int64_t) 512;
        }();
        if (src0->ne[1] < min_rows) {
            return false;
        }
    }
    // dense 2D only; K must be whole K64 stages of whole blocks
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (src0->ne[0] % 256 != 0 || src0->nb[0] != ggml_type_size(src0->type) ||
            src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) {
        return false;
    }
    if (src0->ne[1] > INT_MAX / 2 || src1->ne[1] > INT_MAX / 2) {
        return false;
    }
    return true;
}

// 256 rows x 256 tokens, 8x4 waves (each 32 rows x 64 tokens), gufo's large-projection tile. Best of the tile
// variants measured on gfx1151 (256x256/4x4, 256x256/4x8, 128x256/4x4, 256x128/8x2, 128x128/4x2, 128x512/4x8:
// 2-15 % slower; 128x256/2x4, 256x128/4x2, 128x128/2x2, 256x512/8x4 spill).
template <bool PAIRED>
static void w4a16_launch(const ggml_type type, const char * w, const char * w_up, const short * x, float * y,
        const int n, const int m, const int k, const int64_t nb01, const int64_t stride_y, cudaStream_t stream) {
    constexpr int BM = 256, BN = 256, WM = 8, WN = 4;
    const dim3 grid((n + BN - 1)/BN, (m + (PAIRED ? BM/2 : BM) - 1)/(PAIRED ? BM/2 : BM));
    const dim3 block(WM*WN*32);
    switch (type) {
#define W4A16_CASE(T) case T: mul_mat_w4a16<BM, BN, WM, WN, T, 5, PAIRED><<<grid, block, 0, stream>>>(w, w_up, x, y, n, m, k, nb01, stride_y); break
        W4A16_CASE(GGML_TYPE_IQ4_XS);
        W4A16_CASE(GGML_TYPE_Q4_0);
        W4A16_CASE(GGML_TYPE_Q8_0);
        W4A16_CASE(GGML_TYPE_Q4_K);
        W4A16_CASE(GGML_TYPE_Q5_K);
        W4A16_CASE(GGML_TYPE_Q6_K);
#undef W4A16_CASE
        default: GGML_ABORT("unsupported type");
    }
}

// FP16 activations, [n][k] contiguous
static void w4a16_convert_src1(const ggml_tensor * src1, half * x16, cudaStream_t stream) {
    const int64_t k = src1->ne[0];
    const int64_t n = src1->ne[1];
    if (ggml_is_contiguous(src1)) {
        ggml_get_to_fp16_cuda(GGML_TYPE_F32)(src1->data, x16, n*k, stream);
    } else {
        ggml_get_to_fp16_nc_cuda(GGML_TYPE_F32)(src1->data, x16, k, n, 1, 1,
            src1->nb[1]/sizeof(float), src1->nb[2]/sizeof(float), src1->nb[3]/sizeof(float), stream);
    }
}

void ggml_cuda_mul_mat_w4a16_swiglu(ggml_backend_cuda_context & ctx, const ggml_tensor * w_gate, const ggml_tensor * w_up,
                                    const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(w_gate->type == w_up->type && w_gate->nb[1] == w_up->nb[1] && ggml_are_same_shape(w_gate, w_up));
    static bool logged = false;
    if (!logged) {
        GGML_LOG_INFO("%s: W4A16 fused gate/up/SwiGLU prefill path active (%s, n=%lld)\n", __func__,
            ggml_type_name(w_gate->type), (long long) src1->ne[1]);
        logged = true;
    }
    const int k = w_gate->ne[0];
    const int m = w_gate->ne[1];
    const int n = src1->ne[1];
    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<half> x16(ctx.pool(), (size_t) n*k);
    w4a16_convert_src1(src1, x16.get(), stream);

    const int64_t stride_y = dst->nb[1] / sizeof(float);
    w4a16_launch<true>(w_gate->type, (const char *) w_gate->data, (const char *) w_up->data, (const short *) x16.get(),
        (float *) dst->data, n, m, k, w_gate->nb[1], stride_y, stream);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_mul_mat_w4a16(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    static bool logged = false;
    if (!logged) {
        GGML_LOG_INFO("%s: W4A16 prefill GEMM path active (%s, n=%lld)\n", __func__,
            ggml_type_name(src0->type), (long long) src1->ne[1]);
        logged = true;
    }
    const int k = src0->ne[0];
    const int m = src0->ne[1];
    const int n = src1->ne[1];
    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<half> x16(ctx.pool(), (size_t) n*k);
    w4a16_convert_src1(src1, x16.get(), stream);

    const int64_t stride_y = dst->nb[1] / sizeof(float);
    const char  * w  = (const char *) src0->data;
    const short * xs = (const short *) x16.get();
    float       * yd = (float *) dst->data;
    const int64_t nb01 = src0->nb[1];

    // 256 rows x 256 tokens, 8x4 waves (each 32 rows x 64 tokens). Measured best of 7 tile variants on gfx1151
    // (128x256, 256x128, 128x128 spill; 256x256/4x4, 256x256/4x8, 128x256/4x4 were 3-10 % slower).
    w4a16_launch<false>(src0->type, w, nullptr, xs, yd, n, m, k, nb01, stride_y, stream);
    CUDA_CHECK(cudaGetLastError());
}
