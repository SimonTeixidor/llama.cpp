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
#include "mmid.cuh"

#if defined(GGML_USE_HIP) && defined(RDNA3)
#define W4A16_DEVICE
#endif

// IQ3_XXS weights also take the W4A16 dense / routed kernels (qwen35moe IQ4_XS-4.19bpw: shared experts in 33 layers,
// 6 gate_up + 3 down expert layers, a few projections; on MMQ they ran at ~2 TF). GGML_HIP_W4A16_IQ3=0: back to MMQ.
static bool ggml_cuda_w4a16_iq3() {
    static const bool on = [] {
        const char * e = getenv("GGML_HIP_W4A16_IQ3");
        return e == nullptr || atoi(e) != 0;
    }();
    return on;
}

typedef _Float16 w4a16_h16 __attribute__((ext_vector_type(16)));
typedef float    w4a16_f8  __attribute__((ext_vector_type(8)));

// F32 -> FP16 with round-to-nearest of the already rounded F32 value. Without the empty asm the compiler fuses the
// multiply that produced v with the conversion into one v_fma_mixlo_f16 (single rounding), which is not what the
// unfused path (F32 store, then a separate conversion kernel) computes. `#pragma clang fp contract(off)` does not
// prevent that combine (checked in the ISA), the register barrier does.
static __device__ __forceinline__ half w4a16_f2h_exact(float v) {
    asm volatile("" : "+v"(v));
    return __float2half(v);
}

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
    } else if constexpr (type == GGML_TYPE_IQ3_XXS) {
        // 98-byte blocks (2-byte aligned): d, 64 grid-index bytes, 8 x u32 scale+signs words
        const uint8_t * b = row + (sub/16)*sizeof(block_iq3_xxs);
        const unsigned ib32 = (sub/2) & 7;
        r.h0 = *reinterpret_cast<const uint16_t *>(b);
        const uint16_t * q = reinterpret_cast<const uint16_t *>(b + 2 + 8*ib32 + 4*(sub & 1));
        r.lo[0] = (uint32_t) q[0] | ((uint32_t) q[1] << 16);
        const uint16_t * ss = reinterpret_cast<const uint16_t *>(b + 2 + 64 + 4*ib32);
        r.h1 = (uint32_t) ss[0] | ((uint32_t) ss[1] << 16);
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
    } else if constexpr (type == GGML_TYPE_IQ3_XXS) {
        // ggml dequantize_row_iq3_xxs: db = d*(0.5 + aux>>28)*0.5, 8 values per 7-bit sign index, sign parity bit 7
        const uint32_t aux = r.h1;
        scale = w4a16_h2f(r.h0) * (0.5f + (float) (aux >> 28)) * 0.5f;
        const unsigned l0 = 2*(sub & 1);
#pragma unroll
        for (int g = 0; g < 2; ++g) {
            const unsigned s7    = (aux >> (7*(l0 + g))) & 127;
            const unsigned signs = s7 | ((__popc(s7) & 1) << 7); // == ksigns_iq2xs[s7]
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const uint32_t grid = iq3xxs_grid[(r.lo[0] >> (8*(2*g + h))) & 0xFF];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int v = (grid >> (8*j)) & 0xFF;
                    q[8*g + 4*h + j] = (signs >> (4*h + j)) & 1 ? -v : v;
                }
            }
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
// Epilogue flags (EPI). The accumulators are always FP32; these only select what is stored.
//   W4A16_EPI_F32: y   [token*stride_y   + row] = v                      (the plain GEMM / SwiGLU output)
//   W4A16_EPI_RES: y   [token*stride_y   + row] = v + res[token*stride_r + row]   (GGML_HIP_W4A16_RESADD: the
//                  residual ADD that consumes the GEMM, fused into the store as in gufo's prefill_residual path;
//                  y may alias res exactly: each element is read and written by the same lane)
//   W4A16_EPI_F16: y16 [token*stride_y16 + row] = (half) v               (GGML_HIP_W4A16_F16ACT bit 4: the SwiGLU
//                  output goes straight to the FP16 activations of ffn_down, as gufo keeps activations FP16)
// The RES variant allocates 192 VGPR with 8 spilled (36 B scratch): ISA shows the spills only in the epilogue, the WMMA
// loop is spill-free, and occupancy is 1 workgroup/CU from the 64 KB LDS in every variant (kernel-resources, no GPU).
// Stores of v are bit-identical to the unfused path: v + res is the same F32 add ggml's ADD does, and (half) v is
// the same round-to-nearest conversion the separate F32->F16 kernel does (w4a16_f2h_exact keeps the compiler from
// fusing the last multiply and the conversion).
enum {
    W4A16_EPI_F32 = 1,
    W4A16_EPI_RES = 2,
    W4A16_EPI_F16 = 4,
};

// ROUTED (MUL_MAT_ID): blockIdx.y indexes an (expert, token tile) map built on the device from the expert
// bounds of the compacted slot list (tiles[0] = number of live tiles), blockIdx.x the weight row tile. Token t of
// a tile is compact slot bounds[e] + tile*BN + t; its activation row is ids_src1[slot], its output row ids_dst[slot].
struct w4a16_id_args {
    const int32_t * tiles;
    const int32_t * bounds;
    const int32_t * ids_src1;
    const int32_t * ids_dst;
    int64_t         nb02;     // bytes between experts
};

template <int BM, int BN, int WM, int WN, ggml_type type, int GROUP_SHIFT, bool PAIRED, int EPI, bool ROUTED = false>
__launch_bounds__(WM*WN*32) __global__ void mul_mat_w4a16(
        const char * __restrict__ w, const char * __restrict__ w_up, const short * __restrict__ x, float * y,
        const int n, const int m, const int k, const int64_t w_row_bytes, const int64_t stride_y,
        const float * res, const int64_t stride_r, half * __restrict__ y16, const int64_t stride_y16,
        const w4a16_id_args ida = {nullptr, nullptr, nullptr, nullptr, 0}) {
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
    int e_begin = 0, e_count = n;
    if constexpr (ROUTED) {
        // routed: x = row tile (fastest, so the row tiles of one token tile share its activations in L2)
        if ((int) blockIdx.y >= ida.tiles[0]) {
            return;
        }
        const int tile = ida.tiles[1 + blockIdx.y];
        const int e    = tile & 0xFFFF;
        row_tile   = blockIdx.x;
        token_tile = tile >> 16;
        e_begin    = ida.bounds[e];
        e_count    = ida.bounds[e + 1] - e_begin;
        w += e*ida.nb02;
        if constexpr (PAIRED) {
            w_up += e*ida.nb02;
        }
    } else {
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
        if constexpr (ROUTED) {
            x_row[p] = x + (int64_t) ida.ids_src1[e_begin + (t < e_count ? t : e_count - 1)]*k;
        } else {
            x_row[p] = x + (int64_t) (t < n ? t : n - 1)*k;
        }
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
    // output row of token t of this tile (-1: none)
    const auto out_row = [&](const int token) -> int64_t {
        if constexpr (ROUTED) {
            // routed: token indexes the expert's compact slots
            return token < e_count ? (int64_t) ida.ids_dst[e_begin + token] : -1;
        } else {
            return token < n ? (int64_t) token : -1;
        }
    };
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
                    const int64_t orow = out_row(t0 + 2*l + hi);
                    if (row < m && orow >= 0) {
                        const float g = acc[i][j][l];
                        const float u = acc[i + 1][j][l];
                        const float v = g/(1.0f + expf(-g)) * u;
                        if constexpr (EPI & W4A16_EPI_F32) {
                            y[orow*stride_y + row] = v;
                        }
                        if constexpr (EPI & W4A16_EPI_F16) {
                            y16[orow*stride_y16 + row] = w4a16_f2h_exact(v);
                        }
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
                const int64_t orow = out_row(t0 + 2*l + hi);
                if (row < m && orow >= 0) {
                    const float v = acc[i][j][l];
                    if constexpr (EPI & W4A16_EPI_RES) {
                        y[orow*stride_y + row] = v + res[orow*stride_r + row];
                    } else if constexpr (EPI & W4A16_EPI_F32) {
                        y[orow*stride_y + row] = v;
                    }
                    if constexpr (EPI & W4A16_EPI_F16) {
                        y16[orow*stride_y16 + row] = __float2half(v);
                    }
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(w, w_up, x, y, n, m, k, w_row_bytes, stride_y, res, stride_r, y16, stride_y16, ida);
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
        case GGML_TYPE_IQ3_XXS:
            if (!ggml_cuda_w4a16_iq3()) {
                return false;
            }
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
struct w4a16_out {
    float *       y         = nullptr;
    int64_t       stride_y  = 0;
    const float * res       = nullptr;
    int64_t       stride_r  = 0;
    half *        y16       = nullptr;
    int64_t       stride_16 = 0;
};

template <bool PAIRED, int EPI>
static void w4a16_launch(const ggml_type type, const char * w, const char * w_up, const short * x, const w4a16_out & o,
        const int n, const int m, const int k, const int64_t nb01, cudaStream_t stream) {
    constexpr int BM = 256, BN = 256, WM = 8, WN = 4;
    const dim3 grid((n + BN - 1)/BN, (m + (PAIRED ? BM/2 : BM) - 1)/(PAIRED ? BM/2 : BM));
    const dim3 block(WM*WN*32);
    switch (type) {
#define W4A16_CASE(T) case T: mul_mat_w4a16<BM, BN, WM, WN, T, 5, PAIRED, EPI><<<grid, block, 0, stream>>>( \
        w, w_up, x, o.y, n, m, k, nb01, o.stride_y, o.res, o.stride_r, o.y16, o.stride_16); break
        W4A16_CASE(GGML_TYPE_IQ4_XS);
        W4A16_CASE(GGML_TYPE_Q4_0);
        W4A16_CASE(GGML_TYPE_Q8_0);
        W4A16_CASE(GGML_TYPE_Q4_K);
        W4A16_CASE(GGML_TYPE_Q5_K);
        W4A16_CASE(GGML_TYPE_Q6_K);
        W4A16_CASE(GGML_TYPE_IQ3_XXS);
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

// ---------------------------------------------------------------------------------------------------------------
// GGML_HIP_W4A16_F16ACT: FP16 activations produced once and reused (gufo keeps prefill activations FP16 end to end:
// prefill_norm.hip writes the normed hidden state as FP16 and every projection reads it, prefill_chunk.cpp).
// The F32 tensors are still written (except the SwiGLU output under bit 4, which has exactly one consumer), so any
// cache miss simply falls back to converting F32 again.

int ggml_cuda_w4a16_f16act() {
    static const int v = [] {
        const char * e = getenv("GGML_HIP_W4A16_F16ACT");
        int f = e ? atoi(e) : 7;
        if (f & 2) {
            f |= 1; // the norm writes into the shared cache, so bit 2 implies bit 1
        }
        return f;
    }();
    return ggml_cuda_w4a16_mode() == 1 ? v : 0;
}

bool ggml_cuda_w4a16_resadd() {
    static const bool v = [] {
        const char * e = getenv("GGML_HIP_W4A16_RESADD");
        return e == nullptr || atoi(e) != 0;
    }();
    return ggml_cuda_w4a16_mode() == 1 && v;
}

void ggml_cuda_w4a16_x16_reset(ggml_backend_cuda_context & ctx) {
    ctx.w4a16_x16.release();
}

// Called before every executed node: anything that writes over the cached tensor's memory invalidates the copy.
// Within one graph a live tensor is never overwritten (ggml-alloc only reuses memory after the last consumer), so
// this is a belt-and-braces guard against explicit in-place ops; it costs one range test per node.
void ggml_cuda_w4a16_x16_note_write(ggml_backend_cuda_context & ctx, const ggml_tensor * node) {
    auto & c = ctx.w4a16_x16;
    if (c.key == nullptr || node == c.key || node->data == nullptr) {
        return;
    }
    const char * a0 = (const char *) node->data;
    const char * a1 = a0 + ggml_nbytes(node);
    const char * b0 = (const char *) c.data;
    const char * b1 = b0 + c.nb1*c.ne1;
    if (a0 < b1 && b0 < a1) {
        c.release();
    }
}

static bool w4a16_x16_hit(const ggml_backend_cuda_context & ctx, const ggml_tensor * t) {
    const auto & c = ctx.w4a16_x16;
    return c.key == t && c.buf != nullptr && c.data == t->data && c.ne0 == t->ne[0] && c.ne1 == t->ne[1] && c.nb1 == t->nb[1];
}

// New cache entry for t (evicting the old one first, which keeps the pool usage stack-like), returns its buffer.
static half * w4a16_x16_insert(ggml_backend_cuda_context & ctx, const ggml_tensor * t) {
    auto & c = ctx.w4a16_x16;
    c.release();
    c.pool = &ctx.pool();
    c.buf  = (half *) c.pool->alloc((size_t) t->ne[0]*t->ne[1]*sizeof(half), &c.size);
    c.key  = t;
    c.data = t->data;
    c.ne0  = t->ne[0];
    c.ne1  = t->ne[1];
    c.nb1  = t->nb[1];
    return c.buf;
}

// FP16 [n][k] view of src1: cached copy (bit 1), else a fresh conversion into `local`.
static const half * w4a16_get_x16(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, ggml_cuda_pool_alloc<half> & local) {
    cudaStream_t stream = ctx.stream();
    // the cache is only used on the main stream (graph-level stream concurrency would race on it)
    const bool share = (ggml_cuda_w4a16_f16act() & 1) && ctx.curr_stream_no == 0;
    if (share && w4a16_x16_hit(ctx, src1)) {
        return ctx.w4a16_x16.buf;
    }
    half * x16 = share ? w4a16_x16_insert(ctx, src1) : local.alloc(ctx.pool(), (size_t) src1->ne[0]*src1->ne[1]);
    w4a16_convert_src1(src1, x16, stream);
    return x16;
}

// rms_norm(x)*w -> F32 dst and FP16 copy. Same block size, reduction and expression order as norm.cu's
// rms_norm_f32<1024, true> (ncols >= 1024), so the F32 output is bit-identical to the stock fused RMS_NORM+MUL.
template <int block_size>
static __global__ void w4a16_rms_norm_mul_f16(const float * __restrict__ x, const float * __restrict__ w,
        float * __restrict__ dst, half * __restrict__ dst16, const int ncols, const int64_t stride_row,
        const int64_t stride_dst, const float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    x     += row*stride_row;
    dst   += row*stride_dst;
    dst16 += (int64_t) row*ncols;

    float tmp = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }
    __shared__ float s_sum[32];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);
    for (int col = tid; col < ncols; col += block_size) {
        const float v = scale * x[col] * w[col];
        dst[col]   = v;
        dst16[col] = w4a16_f2h_exact(v);
    }
}

bool ggml_cuda_w4a16_can_norm_f16(const ggml_tensor * rms_norm, const ggml_tensor * mul) {
    if (!(ggml_cuda_w4a16_f16act() & 2) || rms_norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL) {
        return false;
    }
    const ggml_tensor * x = rms_norm->src[0];
    const ggml_tensor * w = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];
    if (w == rms_norm || (mul->src[0] != rms_norm && mul->src[1] != rms_norm)) {
        return false;
    }
    const int64_t k = x->ne[0];
    return x->type == GGML_TYPE_F32 && w->type == GGML_TYPE_F32 && mul->type == GGML_TYPE_F32 &&
        k >= 1024 && ggml_nelements(w) == k && w->ne[0] == k && ggml_is_contiguous(w) &&
        x->nb[0] == sizeof(float) && x->ne[2] == 1 && x->ne[3] == 1 &&
        ggml_are_same_shape(mul, x) && mul->nb[0] == sizeof(float) && mul->nb[1] >= (size_t) k*sizeof(float) &&
        x->ne[1] <= INT_MAX;
}

void ggml_cuda_w4a16_norm_f16(ggml_backend_cuda_context & ctx, const ggml_tensor * rms_norm, ggml_tensor * mul) {
    const ggml_tensor * x = rms_norm->src[0];
    const ggml_tensor * w = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];
    float eps;
    memcpy(&eps, rms_norm->op_params, sizeof(float));
    const int k = x->ne[0];
    const int n = x->ne[1];
    // the kernel writes into a fresh cache entry keyed on the MUL output, which the W4A16 GEMMs then find
    half * x16 = w4a16_x16_insert(ctx, mul);
    w4a16_rms_norm_mul_f16<1024><<<n, 1024, 0, ctx.stream()>>>((const float *) x->data, (const float *) w->data,
        (float *) mul->data, x16, k, x->nb[1]/sizeof(float), mul->nb[1]/sizeof(float), eps);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------------------------------------------

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

    ggml_cuda_pool_alloc<half> local;
    const half * x16 = w4a16_get_x16(ctx, src1, local);

    w4a16_out o;
    o.y        = (float *) dst->data;
    o.stride_y = dst->nb[1] / sizeof(float);
    w4a16_launch<true, W4A16_EPI_F32>(w_gate->type, (const char *) w_gate->data, (const char *) w_up->data,
        (const short *) x16, o, n, m, k, w_gate->nb[1], stream);
    CUDA_CHECK(cudaGetLastError());
}

// GGML_HIP_W4A16_F16ACT bit 4: silu(gate(src1))*up(src1) is written only as FP16 into a scratch buffer, and ffn_down
// reads it directly (no F32 SwiGLU tensor, no conversion). dst = down(...) [+ residual].
void ggml_cuda_mul_mat_w4a16_ffn(ggml_backend_cuda_context & ctx, const ggml_tensor * w_gate, const ggml_tensor * w_up,
                                 const ggml_tensor * src1, const ggml_tensor * w_down, ggml_tensor * dst, const ggml_tensor * residual) {
    GGML_ASSERT(w_gate->type == w_up->type && w_gate->nb[1] == w_up->nb[1] && ggml_are_same_shape(w_gate, w_up));
    GGML_ASSERT(w_down->ne[0] == w_gate->ne[1]);
    static bool logged = false;
    if (!logged) {
        GGML_LOG_INFO("%s: W4A16 FFN chain gate/up/SwiGLU -> FP16 -> down%s active (%s/%s, n=%lld)\n", __func__,
            residual ? " + residual" : "", ggml_type_name(w_gate->type), ggml_type_name(w_down->type), (long long) src1->ne[1]);
        logged = true;
    }
    const int k  = w_gate->ne[0];
    const int ff = w_gate->ne[1];
    const int n  = src1->ne[1];
    const int m  = w_down->ne[1];
    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<half> local;
    const half * x16 = w4a16_get_x16(ctx, src1, local);
    ggml_cuda_pool_alloc<half> h16(ctx.pool(), (size_t) n*ff);

    w4a16_out o1;
    o1.y16       = h16.get();
    o1.stride_16 = ff;
    w4a16_launch<true, W4A16_EPI_F16>(w_gate->type, (const char *) w_gate->data, (const char *) w_up->data,
        (const short *) x16, o1, n, ff, k, w_gate->nb[1], stream);
    CUDA_CHECK(cudaGetLastError());

    w4a16_out o2;
    o2.y        = (float *) dst->data;
    o2.stride_y = dst->nb[1] / sizeof(float);
    if (residual) {
        o2.res      = (const float *) residual->data;
        o2.stride_r = residual->nb[1] / sizeof(float);
        w4a16_launch<false, W4A16_EPI_RES>(w_down->type, (const char *) w_down->data, nullptr, (const short *) h16.get(),
            o2, n, m, ff, w_down->nb[1], stream);
    } else {
        w4a16_launch<false, W4A16_EPI_F32>(w_down->type, (const char *) w_down->data, nullptr, (const short *) h16.get(),
            o2, n, m, ff, w_down->nb[1], stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_mul_mat_w4a16(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
                             const ggml_tensor * residual) {
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

    ggml_cuda_pool_alloc<half> local;
    const half * x16 = w4a16_get_x16(ctx, src1, local);

    w4a16_out o;
    o.y        = (float *) dst->data;
    o.stride_y = dst->nb[1] / sizeof(float);
    // 256 rows x 256 tokens, 8x4 waves (each 32 rows x 64 tokens). Measured best of 7 tile variants on gfx1151
    // (128x256, 256x128, 128x128 spill; 256x256/4x4, 256x256/4x8, 128x256/4x4 were 3-10 % slower).
    if (residual) {
        static bool logged_res = false;
        if (!logged_res) {
            GGML_LOG_INFO("%s: W4A16 residual-add epilogue active (%s, n=%lld)\n", __func__,
                ggml_type_name(src0->type), (long long) n);
            logged_res = true;
        }
        o.res      = (const float *) residual->data;
        o.stride_r = residual->nb[1] / sizeof(float);
        w4a16_launch<false, W4A16_EPI_RES>(src0->type, (const char *) src0->data, nullptr, (const short *) x16, o, n, m, k,
            src0->nb[1], stream);
    } else {
        w4a16_launch<false, W4A16_EPI_F32>(src0->type, (const char *) src0->data, nullptr, (const short *) x16, o, n, m, k,
            src0->nb[1], stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------------------------------------------
// Routed MoE variant (MUL_MAT_ID), on by default (GGML_HIP_W4A16_MOE=0 disables).
// Idea after gufo PR #299 (github.com/gufo-org/gufo, MIT): Flash-Next's routed F16 WMMA expert GEMM
// (src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp, RoutedCompact / RoutedF16GEMMKernel): slots compacted
// by expert, an (expert | tile << 16) map so empty tiles launch nothing, row tiles of one token tile adjacent in
// dispatch order. gufo builds the map on the host from counts read back once per layer; here it is built on the
// device from mm_ids_helper's expert bounds (no sync, CUDA-graph safe), and the GEMM body is the dense W4A16 kernel.

int ggml_cuda_w4a16_moe_mode() {
    static const int mode = [] {
        const char * e = getenv("GGML_HIP_W4A16_MOE");
        return e ? atoi(e) : 1;
    }();
    return mode;
}

// Tile: 128 weight rows x 64 tokens, 4x2 waves. Fastest of 8 configs measured on qwen3.6-35b (128x32, 256x32,
// 64x32 and single-wave-column variants: 1-10 % slower kernel time; results/2026-09-28-moe-w4a16/NOTES.md).
static constexpr int W4A16_ID_BM = 128, W4A16_ID_BN = 64, W4A16_ID_WM = 4, W4A16_ID_WN = 2;

static __global__ void w4a16_id_tiles(const int32_t * __restrict__ bounds, int32_t * __restrict__ tiles,
                                      const int n_experts, const int bn) {
    __shared__ int s[1024];
    const int e   = threadIdx.x;
    const int cnt = e < n_experts ? bounds[e + 1] - bounds[e] : 0;
    const int nt  = (cnt + bn - 1)/bn;
    s[e] = nt;
    __syncthreads();
    for (int off = 1; off < (int) blockDim.x; off <<= 1) {
        const int v = e >= off ? s[e - off] : 0;
        __syncthreads();
        s[e] += v;
        __syncthreads();
    }
    const int start = s[e] - nt;
    for (int j = 0; j < nt; ++j) {
        tiles[1 + start + j] = e | (j << 16);
    }
    if (e == (int) blockDim.x - 1) {
        tiles[0] = s[e];
    }
}

bool ggml_cuda_should_use_w4a16_id(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
                                   const ggml_tensor * dst, int cc) {
    if (ggml_cuda_w4a16_moe_mode() != 1 || !GGML_CUDA_CC_IS_RDNA3_5(cc) || ids == nullptr) {
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
        case GGML_TYPE_IQ3_XXS:
            if (!ggml_cuda_w4a16_iq3()) {
                return false;
            }
            break;
        default:
            return false;
    }
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) {
        return false;
    }
    if (src1->ne[2] < ggml_cuda_w4a16_min_batch()) { // n_tokens
        return false;
    }
    if (src0->ne[0] % 256 != 0 || src0->nb[0] != ggml_type_size(src0->type) || src0->ne[3] != 1 ||
            src1->ne[3] != 1 || src1->nb[0] != sizeof(float) || src1->nb[2] % src1->nb[1] != 0 ||
            dst->nb[0] != sizeof(float) || dst->nb[2] != dst->ne[1]*dst->nb[1]) {
        return false;
    }
    if (src0->ne[2] > 1024 || ids->ne[0] != dst->ne[1] || ids->nb[0] != sizeof(int32_t) ||
            src1->ne[2]*ids->ne[0] >= (1 << 22) || src0->ne[1] > INT_MAX/2) {
        return false;
    }
    return true;
}

template <bool PAIRED>
static void w4a16_id_launch(const ggml_type type, const char * w, const char * w_up, const short * x, float * y,
        const int m, const int k, const int64_t nb01, const int64_t stride_y, const w4a16_id_args & ida,
        const int max_tiles, cudaStream_t stream) {
    constexpr int BM = W4A16_ID_BM, BN = W4A16_ID_BN, WM = W4A16_ID_WM, WN = W4A16_ID_WN;
    const dim3 grid((m + (PAIRED ? BM/2 : BM) - 1)/(PAIRED ? BM/2 : BM), max_tiles);
    const dim3 block(WM*WN*32);
    switch (type) {
#define W4A16_ID_CASE(T) case T: mul_mat_w4a16<BM, BN, WM, WN, T, 0, PAIRED, W4A16_EPI_F32, true><<<grid, block, 0, stream>>>( \
        w, w_up, x, y, 0, m, k, nb01, stride_y, nullptr, 0, nullptr, 0, ida); break
        W4A16_ID_CASE(GGML_TYPE_IQ4_XS);
        W4A16_ID_CASE(GGML_TYPE_Q4_0);
        W4A16_ID_CASE(GGML_TYPE_Q8_0);
        W4A16_ID_CASE(GGML_TYPE_Q4_K);
        W4A16_ID_CASE(GGML_TYPE_Q5_K);
        W4A16_ID_CASE(GGML_TYPE_Q6_K);
        W4A16_ID_CASE(GGML_TYPE_IQ3_XXS);
#undef W4A16_ID_CASE
        default: GGML_ABORT("unsupported type");
    }
}

// src0 [k, m_total, n_expert]; rows [0, m) (or gate rows [0, m) + up rows [m, 2m) when PAIRED) of each expert.
template <bool PAIRED>
static void w4a16_id_run(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                         const ggml_tensor * ids, float * y, const int64_t stride_y, const int m) {
    cudaStream_t stream = ctx.stream();
    const int64_t k        = src0->ne[0];
    const int64_t n_expert = src0->ne[2];
    const int64_t n_used   = ids->ne[0];
    const int64_t n_tokens = src1->ne[2];
    const int64_t ne11     = src1->ne[1];
    const int64_t n_slots  = n_tokens*n_used;

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), n_slots);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), n_slots);
    ggml_cuda_pool_alloc<int32_t> bounds(ctx.pool(), n_expert + 1);
    // activation row of a slot, in units of src1 rows: it*sis1 + iex % ne11 (x16 below is [n_tokens][ne11][k])
    ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), bounds.get(),
        n_expert, n_tokens, n_used, ne11, ids->nb[1]/sizeof(int32_t), ne11, /*write_inverse =*/ false, stream);
    CUDA_CHECK(cudaGetLastError());

    ggml_cuda_pool_alloc<half> x16(ctx.pool(), (size_t) (n_tokens*ne11*k));
    ggml_get_to_fp16_nc_cuda(GGML_TYPE_F32)(src1->data, x16.get(), k, ne11, n_tokens, 1,
        src1->nb[1]/sizeof(float), src1->nb[2]/sizeof(float), src1->nb[3]/sizeof(float), stream);

    const int bn        = W4A16_ID_BN;
    const int max_tiles = (int) (n_slots/bn + std::min<int64_t>(n_expert, n_slots) + 1);
    ggml_cuda_pool_alloc<int32_t> tiles(ctx.pool(), 1 + max_tiles);
    w4a16_id_tiles<<<1, (unsigned) GGML_PAD(n_expert, 32), 0, stream>>>(bounds.get(), tiles.get(), (int) n_expert, bn);
    CUDA_CHECK(cudaGetLastError());

    const w4a16_id_args ida = { tiles.get(), bounds.get(), ids_src1.get(), ids_dst.get(), (int64_t) src0->nb[2] };
    const char * w    = (const char *) src0->data;
    const char * w_up = PAIRED ? w + (int64_t) m*src0->nb[1] : nullptr;
    w4a16_id_launch<PAIRED>(src0->type, w, w_up, (const short *) x16.get(), y, m, k, src0->nb[1], stride_y, ida,
        max_tiles, stream);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_mul_mat_id_w4a16(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                const ggml_tensor * ids, ggml_tensor * dst) {
    static bool logged = false;
    if (!logged) {
        GGML_LOG_INFO("%s: W4A16 routed MoE prefill path active (%s, tokens=%lld)\n", __func__,
            ggml_type_name(src0->type), (long long) src1->ne[2]);
        logged = true;
    }
    w4a16_id_run<false>(ctx, src0, src1, ids, (float *) dst->data, dst->nb[1]/sizeof(float), (int) src0->ne[1]);
}

bool ggml_cuda_can_fuse_w4a16_id_swiglu(const ggml_tensor * gate_up, const ggml_tensor * gate, const ggml_tensor * up,
                                        const ggml_tensor * glu, int cc) {
    static const bool nofuse = getenv("GGML_HIP_W4A16_MOE_NOFUSE") != nullptr && atoi(getenv("GGML_HIP_W4A16_MOE_NOFUSE")) != 0;
    if (ggml_cuda_w4a16_moe_mode() != 1 || nofuse) {
        return false;
    }
    if (gate_up->op != GGML_OP_MUL_MAT_ID || glu->op != GGML_OP_GLU || ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU ||
            ggml_get_op_params_i32(glu, 1) != 0) { // not swapped
        return false;
    }
    if (glu->src[0] != gate || glu->src[1] != up || gate->view_src != gate_up || up->view_src != gate_up) {
        return false;
    }
    const int64_t n_ff = gate_up->ne[0]/2;
    if (gate_up->ne[0] != 2*n_ff || gate->ne[0] != n_ff || up->ne[0] != n_ff || gate->view_offs != 0 ||
            up->view_offs != (size_t) n_ff*sizeof(float) || gate->nb[1] != gate_up->nb[1] || up->nb[1] != gate_up->nb[1] ||
            gate->nb[2] != gate_up->nb[2] || up->nb[2] != gate_up->nb[2] || !ggml_are_same_shape(gate, up)) {
        return false;
    }
    if (!ggml_is_contiguous(glu) || glu->type != GGML_TYPE_F32 || !ggml_are_same_shape(glu, gate)) {
        return false;
    }
    const ggml_tensor * w = gate_up->src[0];
    if (w->ne[1] != 2*n_ff || n_ff % 64 != 0) {
        return false;
    }
    return ggml_cuda_should_use_w4a16_id(w, gate_up->src[1], gate_up->src[2], gate_up, cc);
}

void ggml_cuda_mul_mat_id_w4a16_swiglu(ggml_backend_cuda_context & ctx, const ggml_tensor * gate_up, ggml_tensor * glu) {
    static bool logged = false;
    const ggml_tensor * w = gate_up->src[0];
    if (!logged) {
        GGML_LOG_INFO("%s: W4A16 routed MoE fused gate_up/SwiGLU path active (%s, tokens=%lld)\n", __func__,
            ggml_type_name(w->type), (long long) gate_up->src[1]->ne[2]);
        logged = true;
    }
    w4a16_id_run<true>(ctx, w, gate_up->src[1], gate_up->src[2], (float *) glu->data, glu->nb[1]/sizeof(float),
        (int) (w->ne[1]/2));
}
