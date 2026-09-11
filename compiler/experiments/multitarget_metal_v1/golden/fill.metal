#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct _6
{
    float _m0[8];
};

kernel void metal_static_specimen_kernel(device _6& metal_static_specimen_kernel_arg_0 [[buffer(0)]], uint3 gl_WorkGroupID [[threadgroup_position_in_grid]])
{
    metal_static_specimen_kernel_arg_0._m0[gl_WorkGroupID.y + (gl_WorkGroupID.x * 4u)] = 0.0;
}
