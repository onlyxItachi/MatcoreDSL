// 1. Strict FP reduction (no fast math, unannotated C99 loop)
float test_reduction_strict(int N, const float * __restrict__ A) {
    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        sum += A[i];
    }
    return sum;
}
