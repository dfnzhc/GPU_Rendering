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

#include "optixHello.h"
#include "logger.hpp"

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
typedef SbtRecord<int> MissSbtRecord;

static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    LOG_DEBUG("[{}]: {}", tag, message);
}

int main(int argc, char* argv[])
{
    San::LogSystem logger{};
    
    std::string outfile;
    int width = 512;
    int height = 384;

    try {
        char log[2048]; // For error reporting from OptiX creation functions

        //
        // 初始化 CUDA 以及创建 Optix 上下文
        //
        OptixDeviceContext context = nullptr;
        {
            // Initialize CUDA
            CUDA_CHECK(cudaFree(nullptr));

            CUcontext cuCtx = nullptr;  // zero means take the current context
            OPTIX_CHECK(optixInit());
            OptixDeviceContextOptions options = {};
            options.logCallbackFunction = &context_log_cb;
            options.logCallbackLevel = 4;
            OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));
        }

        //
        // 创建 module
        //
        OptixModule module = nullptr;
        OptixPipelineCompileOptions pipeline_compile_options = {};
        {
            OptixModuleCompileOptions module_compile_options = {};
            module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
            module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
            module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

            pipeline_compile_options.usesMotionBlur = false;
            pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
            pipeline_compile_options.numPayloadValues = 2;
            pipeline_compile_options.numAttributeValues = 2;
            pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;  // TODO: should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
            pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

            size_t inputSize = 0;
            const char* input = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "draw_solid_color.cu", inputSize);

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
        }

        //
        // 创建程序组
        //
        OptixProgramGroup raygen_prog_group = nullptr;
        OptixProgramGroup miss_prog_group = nullptr;
        {
            OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros

            OptixProgramGroupDesc raygen_prog_group_desc = {}; //
            raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            raygen_prog_group_desc.raygen.module = module;
            raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__draw_solid_color";
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

            // 让 miss 模组暂时为空
            OptixProgramGroupDesc miss_prog_group_desc = {};
            miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
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
        }

        //
        // 链接管线
        //
        OptixPipeline pipeline = nullptr;
        {
            const uint32_t max_trace_depth = 0;
            OptixProgramGroup program_groups[] = {raygen_prog_group};

            OptixPipelineLinkOptions pipeline_link_options = {};
            pipeline_link_options.maxTraceDepth = max_trace_depth;
            pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
            size_t sizeof_log = sizeof(log);
            OPTIX_CHECK_LOG(optixPipelineCreate(
                context,
                &pipeline_compile_options,
                &pipeline_link_options,
                program_groups,
                sizeof(program_groups) / sizeof(program_groups[0]),
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
                                                  2  // maxTraversableDepth
            ));
        }

        //
        // 设置 shader binding table(SBT)
        //
        OptixShaderBindingTable sbt = {};
        {
            CUdeviceptr raygen_record;
            const size_t raygen_record_size = sizeof(RayGenSbtRecord);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>( &raygen_record ), raygen_record_size));
            RayGenSbtRecord rg_sbt;
            OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
            rg_sbt.data = {0.462f, 0.725f, 0.f};
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void*>( raygen_record ),
                &rg_sbt,
                raygen_record_size,
                cudaMemcpyHostToDevice
            ));

            CUdeviceptr miss_record;
            size_t miss_record_size = sizeof(MissSbtRecord);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>( &miss_record ), miss_record_size));
            RayGenSbtRecord ms_sbt;
            OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void*>( miss_record ),
                &ms_sbt,
                miss_record_size,
                cudaMemcpyHostToDevice
            ));

            sbt.raygenRecord = raygen_record;
            sbt.missRecordBase = miss_record;
            sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
            sbt.missRecordCount = 1;
        }

        sutil::CUDAOutputBuffer<uchar4> output_buffer(sutil::CUDAOutputBufferType::CUDA_DEVICE, width, height);

        //
        // launch
        //
        {
            CUstream stream;
            CUDA_CHECK(cudaStreamCreate(&stream));

            Params params;
            params.image = output_buffer.map();
            params.image_width = width;

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
        // 显示结果
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
        // 清理
        //
        {
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>( sbt.raygenRecord       )));
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>( sbt.missRecordBase     )));

            OPTIX_CHECK(optixPipelineDestroy(pipeline));
            OPTIX_CHECK(optixProgramGroupDestroy(miss_prog_group));
            OPTIX_CHECK(optixProgramGroupDestroy(raygen_prog_group));
            OPTIX_CHECK(optixModuleDestroy(module));

            OPTIX_CHECK(optixDeviceContextDestroy(context));
        }
    }
    catch (std::exception& e) {
        LOG_ERROR("Caught exception: {}", e.what());
        return 1;
    }
    return 0;
}
