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
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/sutil.h>
#include <sampleConfig.h>

#include "cuda/sphere.h"
#include "sphere.h"
#include "logger.hpp"
#include "sutil/Camera.h"

#include <iomanip>
#include <iostream>
#include <string>

template<typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<MissData> MissSbtRecord;
typedef SbtRecord<sphere::SphereHitGroupData> HitGroupSbtRecord;

void configureCamera(sutil::Camera& cam, const uint32_t width, const uint32_t height)
{
    cam.setEye({0.0f, 0.0f, 3.0f});
    cam.setLookat({0.0f, 0.0f, 0.0f});
    cam.setUp({0.0f, 1.0f, 3.0f});
    cam.setFovY(60.0f);
    cam.setAspectRatio((float) width / (float) height);
}

static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    LOG_DEBUG("[{}]: {}", tag, message);
}

int main(int argc, char* argv[])
{
    San::LogSystem logger{};

    std::string outfile;
    int width = 1920;
    int height = 1080;

    try {
        char log[2048]; // For error reporting from OptiX creation functions

        //
        // ????????? CUDA ???????????? Optix ?????????
        //
        OptixDeviceContext context = nullptr;
        {
            // Initialize CUDA
            CUDA_CHECK(cudaFree(nullptr));

            OPTIX_CHECK(optixInit());

            OptixDeviceContextOptions options = {};
            options.logCallbackFunction = &context_log_cb;
            options.logCallbackLevel = 4;

            CUcontext cuCtx = nullptr;  // zero means take the current context
            OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));
        }

        //
        // ???????????????????????????????????????
        //
        OptixTraversableHandle gasHandle;
        CUdeviceptr d_GasOutputBuffer;
        {
            // ????????????????????????
            OptixAccelBuildOptions accelOptions = {};
            accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
            accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

            // ?????? AABB
            OptixAabb aabb = {-1.5f, -1.5f, -1.5f, 1.5f, 1.5f, 1.5f};
            CUdeviceptr d_aabb_buffer;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>( &d_aabb_buffer ), sizeof(OptixAabb)));
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void*>( d_aabb_buffer ),
                &aabb,
                sizeof(OptixAabb),
                cudaMemcpyHostToDevice
            ));

            // ???????????????????????????????????????????????????????????? IA ????????????
            OptixBuildInput aabbInput = {};
            aabbInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
            aabbInput.customPrimitiveArray.numPrimitives = 1;
            aabbInput.customPrimitiveArray.aabbBuffers = &d_aabb_buffer;

            const uint32_t aabbInputFlags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
            aabbInput.customPrimitiveArray.flags = aabbInputFlags;
            aabbInput.customPrimitiveArray.numSbtRecords = 1;

            // ???????????????????????????????????????
            OptixAccelBufferSizes gasBufferSizes;
            OPTIX_CHECK(optixAccelComputeMemoryUsage(context,
                                                     &accelOptions,
                                                     &aabbInput,
                                                     1,
                                                     &gasBufferSizes));

            // ??????????????????????????????????????????
            CUdeviceptr d_tempBufferGas;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_tempBufferGas), gasBufferSizes.tempSizeInBytes));

            // ????????????????????????????????????????????????
            CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
            auto compactedSizeOffset = roundUp<size_t>(gasBufferSizes.outputSizeInBytes, 8ull);
            // (??????)???????????? 8 ???????????????????????????????????????????????????
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_buffer_temp_output_gas_and_compacted_size),
                compactedSizeOffset + 8
            ));

            // ????????????????????????????????????????????????????????????????????????????????????
            OptixAccelEmitDesc emitProperty = {};
            emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
            emitProperty.result = (CUdeviceptr) ((char*) d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

            OPTIX_CHECK(optixAccelBuild(context,
                                        0,                  // CUDA stream
                                        &accelOptions,
                                        &aabbInput,
                                        1,                  // num build inputs
                                        d_tempBufferGas,
                                        gasBufferSizes.tempSizeInBytes,
                                        d_buffer_temp_output_gas_and_compacted_size,
                                        gasBufferSizes.outputSizeInBytes,
                                        &gasHandle,
                                        &emitProperty,      // emitted property list
                                        1                   // num emitted properties
            ));

            // ?????????????????????
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_tempBufferGas)));
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_aabb_buffer)));

            size_t compacted_gas_size;
            CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void*) emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

            // ????????????????????????????????????????????????????????????????????? handle
            if (compacted_gas_size < gasBufferSizes.outputSizeInBytes) {
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_GasOutputBuffer), compacted_gas_size));

                OPTIX_CHECK(optixAccelCompact(context, 0, gasHandle, d_GasOutputBuffer, compacted_gas_size, &gasHandle));

                CUDA_CHECK(cudaFree((void*) d_buffer_temp_output_gas_and_compacted_size));
            } else {
                // ????????????????????????????????????????????????
                d_GasOutputBuffer = d_buffer_temp_output_gas_and_compacted_size;
            }
        }

        //
        // ?????? module
        //
        OptixModule module = nullptr;
        OptixModule sphere_module = nullptr;
        OptixPipelineCompileOptions pipeline_compile_options = {};
        {
            OptixModuleCompileOptions module_compile_options = {};
            module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
            module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
            module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

            pipeline_compile_options.usesMotionBlur = false;
            pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
            pipeline_compile_options.numPayloadValues = 3;
            pipeline_compile_options.numAttributeValues = sphere::NUM_ATTRIBUTE_VALUES;
#ifdef _DEBUG // ??? optix launches ??????????????? Debug??????????????????????????????????????????
            pipeline_compile_options.exceptionFlags =
                OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
            pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
            pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

            size_t inputSize = 0;
            const char* input = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "draw_sphere.cu", inputSize);

            size_t sizeof_log = sizeof(log);
            OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
                context,
                &module_compile_options,
                &pipeline_compile_options,
                input,
                inputSize,
                log,
                &sizeof_log,
                &module
            ));

            input = sutil::getInputData(nullptr, nullptr, "sphere.cu", inputSize);
            OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
                context,
                &module_compile_options,
                &pipeline_compile_options,
                input,
                inputSize,
                log,
                &sizeof_log,
                &sphere_module
            ));
        }

        //
        // ???????????????
        //
        OptixProgramGroup raygen_prog_group = nullptr;
        OptixProgramGroup miss_prog_group = nullptr;
        OptixProgramGroup hitgroup_prog_group = nullptr;
        {
            OptixProgramGroupOptions program_group_options = {};

            // raygen
            OptixProgramGroupDesc raygen_prog_group_desc = {};
            raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            raygen_prog_group_desc.raygen.module = module;
            raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
            size_t sizeof_log = sizeof(log);
            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                context,
                &raygen_prog_group_desc,
                1,   // num program groups
                &program_group_options,
                log,
                &sizeof_log,
                &raygen_prog_group
            ));

            // miss
            OptixProgramGroupDesc miss_prog_group_desc = {};
            miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
            miss_prog_group_desc.miss.module = module;
            miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
            sizeof_log = sizeof(log);
            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                context,
                &miss_prog_group_desc,
                1,   // num program groups
                &program_group_options,
                log,
                &sizeof_log,
                &miss_prog_group
            ));

            // hitgroup
            OptixProgramGroupDesc hitgroup_prog_group_desc = {};
            hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            hitgroup_prog_group_desc.hitgroup.moduleCH = module;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
            hitgroup_prog_group_desc.hitgroup.moduleAH = nullptr;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;
            hitgroup_prog_group_desc.hitgroup.moduleIS = sphere_module;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__sphere";
            sizeof_log = sizeof(log);
            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                context,
                &hitgroup_prog_group_desc,
                1,   // num program groups
                &program_group_options,
                log,
                &sizeof_log,
                &hitgroup_prog_group
            ));
        }

        //
        // ????????????
        //
        OptixPipeline pipeline = nullptr;
        {
            const uint32_t max_trace_depth = 1;
            OptixProgramGroup program_groups[] = {raygen_prog_group, miss_prog_group, hitgroup_prog_group};

            OptixPipelineLinkOptions pipeline_link_options = {};
            pipeline_link_options.maxTraceDepth = max_trace_depth;
            pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
            size_t sizeof_log = sizeof(log);
            OPTIX_CHECK_LOG(optixPipelineCreate(
                context,
                &pipeline_compile_options,
                &pipeline_link_options,
                program_groups,
                std::size(program_groups),
                log,
                &sizeof_log,
                &pipeline
            ));

            OptixStackSizes stack_sizes = {};
            for (auto& prog_group : program_groups) {
                OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
            }

            uint32_t direct_callable_stack_size_from_traversal;
            uint32_t direct_callable_stack_size_from_state;
            uint32_t continuation_stack_size;
            OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
                                                   0,  // maxCCDepth
                                                   0,  // maxDCDEpth
                                                   &direct_callable_stack_size_from_traversal,
                                                   &direct_callable_stack_size_from_state, &continuation_stack_size));
            OPTIX_CHECK(optixPipelineSetStackSize(pipeline, direct_callable_stack_size_from_traversal,
                                                  direct_callable_stack_size_from_state, continuation_stack_size,
                                                  1  // maxTraversableDepth
            ));
        }

        //
        // ?????? shader binding table(SBT)
        //
        OptixShaderBindingTable sbt = {};
        {
            CUdeviceptr raygen_record;
            const size_t raygen_record_size = sizeof(RayGenSbtRecord);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>( &raygen_record ), raygen_record_size));
            sutil::Camera camera;
            configureCamera(camera, width, height);
            RayGenSbtRecord rg_sbt;
            rg_sbt.data = {};
            rg_sbt.data.cam_eye = camera.eye();
            camera.UVWFrame(rg_sbt.data.camera_u, rg_sbt.data.camera_v, rg_sbt.data.camera_w);
            OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void*>( raygen_record ),
                &rg_sbt,
                raygen_record_size,
                cudaMemcpyHostToDevice
            ));

            CUdeviceptr miss_record;
            size_t miss_record_size = sizeof(MissSbtRecord);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>( &miss_record ), miss_record_size));
            MissSbtRecord ms_sbt;
            ms_sbt.data = {0.2f, 0.3f, 0.5f};
            OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void*>( miss_record ),
                &ms_sbt,
                miss_record_size,
                cudaMemcpyHostToDevice
            ));

            CUdeviceptr hitgroup_record;
            size_t hitgroup_record_size = sizeof(HitGroupSbtRecord);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroup_record), hitgroup_record_size));
            HitGroupSbtRecord hg_sbt;
            hg_sbt.data.sphere.center = { 0.0f, 0.0f, 0.0f };
            hg_sbt.data.sphere.radius = 1.5f;
            OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt));
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void*>(hitgroup_record),
                &hg_sbt,
                hitgroup_record_size,
                cudaMemcpyHostToDevice
            ));

            sbt.raygenRecord = raygen_record;
            sbt.missRecordBase = miss_record;
            sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
            sbt.missRecordCount = 1;
            sbt.hitgroupRecordBase = hitgroup_record;
            sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
            sbt.hitgroupRecordCount = 1;
        }

        sutil::CUDAOutputBuffer<uchar4> output_buffer(sutil::CUDAOutputBufferType::CUDA_DEVICE, width, height);

        //
        // launch
        //
        {
            CUstream stream;
            CUDA_CHECK(cudaStreamCreate(&stream));


            // ?????????????????????
            Params params;
            params.image = output_buffer.map();
            params.image_width = width;
            params.image_height = height;
            params.handle = gasHandle;
            params.origin_x = width / 2;
            params.origin_y = height / 2;

            CUdeviceptr d_param;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>( &d_param ), sizeof(Params)));
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void*>( d_param ),
                &params, sizeof(params),
                cudaMemcpyHostToDevice
            ));

            OPTIX_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(Params), &sbt, width, height, /*depth=*/1));
            CUDA_SYNC_CHECK();

            output_buffer.unmap();
        }

        //
        // ????????????
        //
        {
            sutil::ImageBuffer buffer;
            buffer.data = output_buffer.getHostPointer();
            buffer.width = width;
            buffer.height = height;
            buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
            if (outfile.empty())
                sutil::displayBufferWindow(argv[0], buffer);
            else
                sutil::saveImage(outfile.c_str(), buffer, false);
        }

        //
        // ??????
        //
        {
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>( sbt.raygenRecord       )));
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>( sbt.missRecordBase     )));
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>( sbt.hitgroupRecordBase     )));
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>( d_GasOutputBuffer     )));

            OPTIX_CHECK(optixPipelineDestroy(pipeline));
            OPTIX_CHECK(optixProgramGroupDestroy(miss_prog_group));
            OPTIX_CHECK(optixProgramGroupDestroy(raygen_prog_group));
            OPTIX_CHECK(optixProgramGroupDestroy(hitgroup_prog_group));
            OPTIX_CHECK(optixModuleDestroy(module));
            OPTIX_CHECK(optixModuleDestroy(sphere_module));

            OPTIX_CHECK(optixDeviceContextDestroy(context));
        }
    }
    catch (std::exception& e) {
        LOG_ERROR("Caught exception: {}", e.what());
        return 1;
    }

    return 0;
}
