#pragma once

#include "common.cuh"

// W4A16 prefill GEMM for RDNA3.5: quantized weights decoded in-kernel to scaled FP16, FP16 activations,
// FP32 accumulation via v_wmma_f32_16x16x16_f16. On by default on RDNA3.5 for ne11 >= GGML_HIP_W4A16_MIN_BATCH;
// GGML_HIP_W4A16_PREFILL=0 turns it off.
// Adapted from gufo (github.com/gufo-org/gufo, MIT), src/models/qwen/hip/kernels/prefill_fp16.hip.

bool ggml_cuda_should_use_w4a16(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);

// dst = src0 x src1 [+ residual]. residual (GGML_HIP_W4A16_RESADD) is the other operand of the ADD that consumed the
// matmul; dst is then that ADD's tensor. Same shape as dst, F32, rows contiguous; may alias dst exactly.
void ggml_cuda_mul_mat_w4a16(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
                             const ggml_tensor * residual = nullptr);

// dst = silu(gate(src1)) * up(src1), gate/up weights of the same type and shape.
void ggml_cuda_mul_mat_w4a16_swiglu(ggml_backend_cuda_context & ctx, const ggml_tensor * w_gate, const ggml_tensor * w_up,
                                    const ggml_tensor * src1, ggml_tensor * dst);

// Prefill fusions on top of the W4A16 GEMM (all on by default; see results/2026-09-28-gap-fusions/NOTES.md).
//
// GGML_HIP_W4A16_F16ACT (bitmask, default 7; 0 disables):
//   1  share one F32->FP16 conversion of an activation tensor across all W4A16 GEMMs that read it in a graph
//      (qwen35 GDN: attn_qkv/attn_gate/ssm_beta/ssm_alpha; attention: q/k/v)
//   2  RMS_NORM+MUL writes the FP16 copy itself (implies 1): the GEMMs after the norm convert nothing
//   4  FFN chain: gate/up/SwiGLU writes FP16 only, ffn_down reads it (no F32 SwiGLU tensor, no conversion)
// GGML_HIP_W4A16_RESADD (default 1; 0 disables): a residual ADD consuming a W4A16 GEMM output is done in the GEMM's store.
int  ggml_cuda_w4a16_f16act();
bool ggml_cuda_w4a16_resadd();

// FP16 activation cache bookkeeping, called by the graph evaluator.
void ggml_cuda_w4a16_x16_reset(ggml_backend_cuda_context & ctx);
void ggml_cuda_w4a16_x16_note_write(ggml_backend_cuda_context & ctx, const ggml_tensor * node);

bool ggml_cuda_w4a16_can_norm_f16(const ggml_tensor * rms_norm, const ggml_tensor * mul);
void ggml_cuda_w4a16_norm_f16(ggml_backend_cuda_context & ctx, const ggml_tensor * rms_norm, ggml_tensor * mul);

// dst = down(silu(gate(src1)) * up(src1)) [+ residual], the SwiGLU output only ever exists as FP16.
void ggml_cuda_mul_mat_w4a16_ffn(ggml_backend_cuda_context & ctx, const ggml_tensor * w_gate, const ggml_tensor * w_up,
                                 const ggml_tensor * src1, const ggml_tensor * w_down, ggml_tensor * dst, const ggml_tensor * residual);
