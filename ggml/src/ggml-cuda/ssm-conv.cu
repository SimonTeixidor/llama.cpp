#include "common.cuh"
#include "ssm-conv.cuh"
#include "unary.cuh"

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float * src0_ptr, const float * src1_ptr,
                                    const float * bias_ptr,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * dst_ptr, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float * GGML_CUDA_RESTRICT bias = bias_ptr;
    float       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

    ggml_cuda_pdl_sync();
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const float * __restrict__ bias,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const float * bias, const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_conv_f32<apply_silu, threads, kNC>, launch_params, src0, src1, bias, src0_nb0, src0_nb1,
                                                                        src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        } else {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                src0, src1, bias, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 9, 15 right now.");
    }
}

// ggml_ssm_conv_rs: one thread per channel reads its d_conv-1 conv-state values from
// cache row ids[0] and the new tokens from x (any strides), writes the convolution (+ SiLU) and the conv-state snapshot
// windows. Replaces conv GET_ROWS + CONCAT + snapshot CPYs + SSM_CONV. The convolution is spelled as the fma chain from 0
// that ssm_conv_f32 compiles to (as in gdn_decode_fused_cuda), plus the zero bias, so the output is bit-identical.
// In place: every thread reads its own state values into registers before it writes any snapshot.
template <int d_conv, bool apply_silu>
static __global__ void ssm_conv_rs_f32(const char * __restrict__ x, const int64_t x_nb0, const int64_t x_nb1,
                                       const float * __restrict__ w, const int64_t w_stride,
                                       float * cache, const int64_t row_stride, const int32_t * __restrict__ ids,
                                       const int n_slots, const int64_t rs_size, const int64_t row_out,
                                       float * __restrict__ dst, const int64_t dst_stride, const int n_t, const int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) {
        return;
    }
    constexpr int ns = d_conv - 1;

    ggml_cuda_pdl_sync();

    const float * st = cache + (int64_t) ids[0] * row_stride + (int64_t) c * ns;
    float old[ns];
#pragma unroll
    for (int j = 0; j < ns; ++j) {
        old[j] = st[j];
    }
    float wr[d_conv];
#pragma unroll
    for (int j = 0; j < d_conv; ++j) {
        wr[j] = w[(int64_t) c * w_stride + j];
    }

    const char * xc = x + (int64_t) c * x_nb1;

    float win[d_conv];
#pragma unroll
    for (int j = 0; j < ns; ++j) {
        win[j] = old[j];
    }
    for (int i = 0; i < n_t; ++i) {
        win[d_conv - 1] = *(const float *) (xc + (int64_t) i * x_nb0);
        float sumf = fmaf(win[0], wr[0], 0.0f);
#pragma unroll
        for (int j = 1; j < d_conv; ++j) {
            sumf = fmaf(win[j], wr[j], sumf);
        }
        sumf += 0.0f; // the (absent) bias of ssm_conv_f32
        dst[(int64_t) i * dst_stride + c] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
#pragma unroll
        for (int j = 0; j < d_conv - 1; ++j) {
            win[j] = win[j + 1];
        }
    }

    // snapshot slot s: columns max(0, n_t - s) .. + ns - 1 of [old | x]
    for (int s = 0; s < n_slots; ++s) {
        const int start = max(0, n_t - s);
        float * out = cache + ((int64_t) s * rs_size + row_out) * row_stride + (int64_t) c * ns;
#pragma unroll
        for (int j = 0; j < ns; ++j) {
            const int col = start + j;
            out[j] = col < ns ? old[col] : *(const float *) (xc + (int64_t) (col - ns) * x_nb0);
        }
    }
}

static void ggml_cuda_op_ssm_conv_rs(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * silu_dst) {
    const ggml_tensor * x     = dst->src[0];
    const ggml_tensor * w     = dst->src[1];
    const ggml_tensor * cache = dst->src[2];
    const ggml_tensor * ids   = dst->src[3];
    const ggml_tensor * out   = silu_dst != nullptr ? silu_dst : dst;

    const int     d_conv  = (int) w->ne[0];
    const int     C       = (int) w->ne[1];
    const int     n_t     = (int) x->ne[0];
    const int     n_slots = ggml_get_op_params_i32(dst, 1);
    const int64_t rs_size = ggml_get_op_params_i32(dst, 2);
    const int64_t row_out = ggml_get_op_params_i32(dst, 3);

    GGML_ASSERT(out->type == GGML_TYPE_F32 && out->nb[0] == sizeof(float) && out->ne[0] == C && out->ne[1] == n_t);
    GGML_ASSERT(w->nb[0] == sizeof(float) && cache->nb[0] == sizeof(float) && ids->type == GGML_TYPE_I32);

    const int threads = 128;
    const dim3 blocks((C + threads - 1) / threads, 1, 1);
    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(blocks, threads, 0, ctx.stream());

    auto launch = [&](auto DC, auto SILU) {
        constexpr int  kd = decltype(DC)::value;
        constexpr bool ks = decltype(SILU)::value;
        ggml_cuda_kernel_launch(ssm_conv_rs_f32<kd, ks>, lp,
            (const char *) x->data, (int64_t) x->nb[0], (int64_t) x->nb[1],
            (const float *) w->data, (int64_t) (w->nb[1] / sizeof(float)),
            (float *) cache->data, (int64_t) (cache->nb[1] / sizeof(float)), (const int32_t *) ids->data,
            n_slots, rs_size, row_out, (float *) out->data, (int64_t) (out->nb[1] / sizeof(float)), n_t, C);
    };
    GGML_ASSERT(d_conv == 4 && "ggml_ssm_conv_rs: only d_conv == 4");
    if (silu_dst != nullptr) {
        launch(std::integral_constant<int, 4>{}, std::true_type{});
    } else {
        launch(std::integral_constant<int, 4>{}, std::false_type{});
    }
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node, ggml_tensor * silu_dst) {
    if (ggml_get_op_params_i32(dst, 0) == 1) {
        GGML_ASSERT(bias_add_node == nullptr);
        ggml_cuda_op_ssm_conv_rs(ctx, dst, silu_dst);
        return;
    }

    const struct ggml_tensor * src0 = dst->src[0];  // conv_x
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
    const bool fuse_bias = bias_add_node != nullptr;
    const bool fuse_silu = silu_dst != nullptr;

    // bias always comes with silu.
    GGML_ASSERT(!fuse_bias || fuse_silu);

    // The bias (when fused) is the non-conv operand of the ADD node.
    const struct ggml_tensor * bias = fuse_bias ? (bias_add_node->src[0] == dst ? bias_add_node->src[1] : bias_add_node->src[0]) : nullptr;

    // When fusing, write to silu_dst (the node downstream references).
    const struct ggml_tensor * out = fuse_silu ? silu_dst : dst;

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = out->ne[1];                 // tokens per sequence
    const int64_t n_s = out->ne[2];                 // number of sequences in the batch

    GGML_ASSERT(out->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * bias_d = fuse_bias ? (const float *) bias->data : nullptr;
    float *       dst_d  = (float *) out->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);
    if (fuse_bias) {
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(ggml_nelements(bias) == nr);
    }

    if (fuse_silu) {
        ssm_conv_f32_cuda<true>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    } else {
        ssm_conv_f32_cuda<false>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    }
}
