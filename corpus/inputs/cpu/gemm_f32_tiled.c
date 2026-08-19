/* Diagnostic Case: gemm_f32_tiled.c
 * Purpose: Blocked/Tiled 32x32x64 loop GEMM evaluating vectorization and register tiling.
 */
#define TILE_M 32
#define TILE_N 32
#define TILE_K 64

void gemm_f32_tiled(
    int M, int N, int K,
    const float * __restrict__ A,
    const float * __restrict__ B,
    float * __restrict__ C) {
  for (int bi = 0; bi < M; bi += TILE_M) {
    for (int bj = 0; bj < N; bj += TILE_N) {
      for (int bk = 0; bk < K; bk += TILE_K) {
        for (int i = 0; i < TILE_M; ++i) {
          for (int k = 0; k < TILE_K; ++k) {
            float a_val = A[(bi + i) * K + (bk + k)];
            #pragma clang loop vectorize(enable)
            for (int j = 0; j < TILE_N; ++j) {
              C[(bi + i) * N + (bj + j)] += a_val * B[(bk + k) * N + (bj + j)];
            }
          }
        }
      }
    }
  }
}
