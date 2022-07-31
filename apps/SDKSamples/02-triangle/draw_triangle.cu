//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <optix.h>

#include "triangle.h"
#include <cuda/helpers.h>

extern "C" {
__constant__ Params params;
}

static __forceinline__ __device__ void setPayload(float3 p)
{
    optixSetPayload_0(float_as_int(p.x));
    optixSetPayload_1(float_as_int(p.y));
    optixSetPayload_2(float_as_int(p.z));
}

static __forceinline__ __device__ void computeRay(uint3 idx, uint3 dim, float3& origin, float3& dir)
{
    const float3 U = params.cam_u;
    const float3 V = params.cam_v;
    const float3 W = params.cam_w;
    const float2 d = 2.0 * make_float2(static_cast<float>(idx.x) / static_cast<float>(dim.x),
                                       static_cast<float>(idx.y) / static_cast<float>(dim.y)) - 1.0f;

    origin = params.camera_pos;
    dir = normalize(d.x * U + d.y * V + W);
}

extern "C"
__global__ void __raygen__triangle()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // 按照所映射的索引，计算光线
    float3 rayOrigin, rayDir;
    computeRay(idx, dim, rayOrigin, rayDir);

    // 向场景射出光线
    unsigned int p0, p1, p2;
    optixTrace(params.handle,
               rayOrigin,
               rayDir,
               0.0,
               1e16f,
               0.0f,
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_NONE,
               0,
               1,
               0,
               p0, p1, p2);

    float3 result;
    result.x = int_as_float(p0);
    result.y = int_as_float(p1);
    result.z = int_as_float(p2);

    params.image[idx.y * params.image_width + idx.x] = make_color(result);
}

extern "C"
__global__ void __miss__triangle()
{
    auto* missData = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    setPayload(missData->bg_color);
}

extern "C"
__global__ void __closesthit__triangle()
{
    // Optix 有一些内置的属性
    const float2 barycentrics = optixGetTriangleBarycentrics();

    setPayload(make_float3(barycentrics, 1.0f));
}