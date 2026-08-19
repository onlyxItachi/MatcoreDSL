// Diagnostic Case: gemm_cuda_baseline.cu
// Purpose: Public warp-level CUDA GEMM structure mapping to sm_89 / Tensor Core instructions.
#if defined(__CUDACC__) || defined(__CUDA__)
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

__global__ void wmma_gemm_f16_f32(
    const half * __restrict__ a,
    const half * __restrict__ b,
    float * __restrict__ c,
    int M, int N, int K) {
  // Tile dimensions for m16n16k16 wmma
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

  wmma::fill_fragment(c_frag, 0.0f);

  for (int k = 0; k < K; k += 16) {
    wmma::load_matrix_sync(a_frag, a + (blockIdx.y * 16) * K + k, K);
    wmma::load_matrix_sync(b_frag, b + k * N + (blockIdx.x * 16), N);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

  wmma::store_matrix_sync(c + (blockIdx.y * 16) * N + (blockIdx.x * 16), c_frag, N, wmma::mem_row_major);
}
#endif
