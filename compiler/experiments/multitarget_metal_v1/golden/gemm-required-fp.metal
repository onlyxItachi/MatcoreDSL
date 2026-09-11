#pragma clang diagnostic ignored "-Wmissing-prototypes"

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

template<typename T>
[[clang::optnone]] T spvFMul(T l, T r)
{
    return fma(l, r, T(0));
}

template<typename T, int Cols, int Rows>
[[clang::optnone]] vec<T, Cols> spvFMulVectorMatrix(vec<T, Rows> v, matrix<T, Cols, Rows> m)
{
    vec<T, Cols> res = vec<T, Cols>(0);
    for (uint i = Rows; i > 0; --i)
    {
        vec<T, Cols> tmp(0);
        for (uint j = 0; j < Cols; ++j)
        {
            tmp[j] = m[j][i - 1];
        }
        res = fma(tmp, vec<T, Cols>(v[i - 1]), res);
    }
    return res;
}

template<typename T, int Cols, int Rows>
[[clang::optnone]] vec<T, Rows> spvFMulMatrixVector(matrix<T, Cols, Rows> m, vec<T, Cols> v)
{
    vec<T, Rows> res = vec<T, Rows>(0);
    for (uint i = Cols; i > 0; --i)
    {
        res = fma(m[i - 1], vec<T, Rows>(v[i - 1]), res);
    }
    return res;
}

template<typename T, int LCols, int LRows, int RCols, int RRows>
[[clang::optnone]] matrix<T, RCols, LRows> spvFMulMatrixMatrix(matrix<T, LCols, LRows> l, matrix<T, RCols, RRows> r)
{
    matrix<T, RCols, LRows> res;
    for (uint i = 0; i < RCols; i++)
    {
        vec<T, RCols> tmp(0);
        for (uint j = 0; j < LCols; j++)
        {
            tmp = fma(vec<T, RCols>(r[i][j]), l[j], tmp);
        }
        res[i] = tmp;
    }
    return res;
}

template<typename T>
[[clang::optnone]] T spvFAdd(T l, T r)
{
    return fma(T(1), l, r);
}

struct _9
{
    float _m0[6];
};

struct _11
{
    float _m0[12];
};

struct _13
{
    float _m0[8];
};

kernel void metal_static_specimen_kernel(device _9& metal_static_specimen_kernel_arg_0 [[buffer(0)]], device _11& metal_static_specimen_kernel_arg_1 [[buffer(1)]], device _13& metal_static_specimen_kernel_arg_2 [[buffer(2)]], uint3 gl_WorkGroupID [[threadgroup_position_in_grid]])
{
    for (uint _38 = 0u; int(_38) < int(3u); )
    {
        metal_static_specimen_kernel_arg_2._m0[gl_WorkGroupID.y + (gl_WorkGroupID.x * 4u)] = spvFAdd(metal_static_specimen_kernel_arg_2._m0[gl_WorkGroupID.y + (gl_WorkGroupID.x * 4u)], spvFMul(metal_static_specimen_kernel_arg_0._m0[_38 + (gl_WorkGroupID.x * 3u)], metal_static_specimen_kernel_arg_1._m0[gl_WorkGroupID.y + (_38 * 4u)]));
        _38++;
        continue;
    }
}
