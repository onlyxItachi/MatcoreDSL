#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct _6
{
    float _m0[6];
};

struct _12
{
    float _m0[12];
};

struct _17
{
    float _m0[8];
};

kernel void metal_static_specimen_kernel(device _6& metal_static_specimen_kernel_arg_0 [[buffer(0)]], device _12& metal_static_specimen_kernel_arg_1 [[buffer(1)]], device _17& metal_static_specimen_kernel_arg_2 [[buffer(2)]], uint3 gl_WorkGroupID [[threadgroup_position_in_grid]])
{
    for (uint _35 = 0u; int(_35) < int(3u); )
    {
        metal_static_specimen_kernel_arg_2._m0[gl_WorkGroupID.y + (gl_WorkGroupID.x * 4u)] += (metal_static_specimen_kernel_arg_0._m0[_35 + (gl_WorkGroupID.x * 3u)] * metal_static_specimen_kernel_arg_1._m0[gl_WorkGroupID.y + (_35 * 4u)]);
        _35++;
        continue;
    }
}
