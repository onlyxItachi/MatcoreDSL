// Diagnostic Case: gemm_hip_baseline.hip
// Purpose: Public wave-level HIP GEMM structure mapping to AMD gfx9 / gfx11 Matrix Cores.
#if defined(__HIPCC__) || defined(__HIP__)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__global__ void hip_gemm_naive_wave(
    const half * __restrict__ a,
    const half * __restrict__ b,
    float * __restrict__ c,
    int M, int N, int K) {
  int row = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
  int col = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

  if (row < M && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
      sum += __half2float(a[row * K + k]) * __half2float(b[k * N + col]);
    }
    c[row * N + col] = sum;
  }
}
#endif
