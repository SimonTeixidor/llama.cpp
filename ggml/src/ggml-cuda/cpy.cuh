#include "common.cuh"

#define CUDA_CPY_BLOCK_SIZE 64

void ggml_cuda_cpy(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ggml_tensor * src1);

void ggml_cuda_dup(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#define GGML_CUDA_CPY_BATCH_MAX 32
#define CUDA_CPY_BLOCK_SIZE_BATCHED 256

struct ggml_cuda_cpy_batch_offs {
    int64_t src[GGML_CUDA_CPY_BATCH_MAX];
    int64_t dst[GGML_CUDA_CPY_BATCH_MAX];
};

bool ggml_cuda_cpy_batched_supported(const ggml_tensor * src0, const ggml_tensor * src1);

// srcs[i] -> dsts[i]; all copies must share type, ne and nb (checked by the caller)
void ggml_cuda_cpy_batched(ggml_backend_cuda_context & ctx, const ggml_tensor * const * srcs, const ggml_tensor * const * dsts, int n);
