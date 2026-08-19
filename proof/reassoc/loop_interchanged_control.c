// 3. Loop interchanged parallel accumulation (no horizontal reduction needed)
void test_reduction_interchanged(int N, int M, const float * __restrict__ A, float * __restrict__ sum) {
    for (int j = 0; j < M; ++j) sum[j] = 0.0f;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            sum[j] += A[i * M + j];
        }
    }
}
