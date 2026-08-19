// 2. Reassociation enabled reduction (compiled with -fassociative-math or fast-math)
float test_reduction_reassoc(int N, const float * __restrict__ A) {
    float sum = 0.0f;
    #pragma clang loop vectorize(enable)
    for (int i = 0; i < N; ++i) {
        sum += A[i];
    }
    return sum;
}
