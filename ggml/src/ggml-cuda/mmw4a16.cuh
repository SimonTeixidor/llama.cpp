#pragma once

#include "common.cuh"

// W4A16 prefill GEMM for RDNA3.5: quantized weights decoded in-kernel to scaled FP16, FP16 activations,
// FP32 accumulation via v_wmma_f32_16x16x16_f16. On by default on RDNA3.5 for ne11 >= GGML_HIP_W4A16_MIN_BATCH;
// GGML_HIP_W4A16_PREFILL=0 turns it off.
// Adapted from gufo (github.com/gufo-org/gufo, MIT), src/models/qwen/hip/kernels/prefill_fp16.hip.

bool ggml_cuda_should_use_w4a16(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);

void ggml_cuda_mul_mat_w4a16(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// dst = silu(gate(src1)) * up(src1), gate/up weights of the same type and shape.
void ggml_cuda_mul_mat_w4a16_swiglu(ggml_backend_cuda_context & ctx, const ggml_tensor * w_gate, const ggml_tensor * w_up,
                                    const ggml_tensor * src1, ggml_tensor * dst);
