/*
* Copyright (c) 2020-2021 Arm Limited.
*
* SPDX-License-Identifier: MIT
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to
* deal in the Software without restriction, including without limitation the
* rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
* sell copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/core/WindowIterator.h"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

#include "support/ToolchainSupport.h"
#include "utils/Utils.h"

#include <cstdlib>
#include <chrono>
#include <sys/sysctl.h>

//#define enableTransferTensorsWithoutQuantINfo2 0

using namespace arm_compute;
using namespace utils;

void check_system_isa(const std::string& str) {
    int64_t ret = 0;
    size_t size = sizeof(ret);
    if (sysctlbyname(str.data(), &ret, &size, NULL, 0) == 0) {
        std::cout << "--------------" << std::endl;
        std::cout << " Exist in system: " << str << std::endl;
        std::cout << "--------------" << std::endl;
    }
}

void print_results(int iter_count, const std::function<void()>& perf_func, const std::string& test_name) {
    std::vector<double> time_list(iter_count, 0.0);
    for (int i = 0; i < iter_count; i++) {
        auto m_StartTime = std::chrono::system_clock::now();
        perf_func();
        auto m_EndTime = std::chrono::system_clock::now();
        time_list[i] = std::chrono::duration_cast<std::chrono::microseconds>(m_EndTime - m_StartTime).count();
    }

    std::sort(time_list.begin(), time_list.end());
    std::cout << test_name << " median time = " << time_list[iter_count / 2] << " microsecs." << std::endl;
    std::cout << test_name << " average time = ";
    std::cout << accumulate(time_list.begin(), time_list.end(), 0.0) / time_list.size() << " microsecs." << std::endl;
}

// Find min and max value in a float array
void find_min_max(int size, const float *data, float *min, float *max)
{
   *min = *max = data[0];
   for (int i = 0; i < size; i++)
   {
       const float val = data[i];
       *min            = std::min(*min, val);
       *max            = std::max(*max, val);
   }
}

// Return reasonable quantisation parameters to use for an array of floats
// based on min and max values
QuantizationInfo choose_quantization_params(float min, float max)
{
   // Extend the [min,max] interval to contain 0 so we can represent it exactly
   min = std::min(min, 0.f);
   max = std::max(max, 0.f);

   // Set the quantized min and max in float values
   const float qmin = 0;
   const float qmax = 255;

   // Determine the scale
   const float scale = (max - min) / (qmax - qmin);

   // Determine the zero-point; using affine equation val = (qval-zerop) * scale
   const float zero_point_real = qmin - min / scale;

   // But we need to nudge the zero_point to an integer (exact quantized value)
   std::uint8_t zero_point_nudged = 0;
   if (zero_point_real < qmin)
   {
       zero_point_nudged = qmin;
   }
   else if (zero_point_real > qmax)
   {
       zero_point_nudged = qmax;
   }
   else
   {
       zero_point_nudged = static_cast<std::uint8_t>(support::cpp11::round(zero_point_real));
   }

   QuantizationInfo qinfo = QuantizationInfo(scale, zero_point_nudged);
   return qinfo;
}

void quantize_values(int size, qasymm8_t *output, float *input, const QuantizationInfo qinfo)
{
   for (int i = 0; i < size; i++)
   {
       output[i] = quantize_qasymm8(input[i], qinfo);
   }
   std::cout << "\n";
}

int main(int argc, char **argv)
{
    check_system_isa("hw.optional.arm.FEAT_FHM");
    const int count_iter = 1000;
    Tensor    src1, src1_f16;
    Tensor    src2, src2_f16;
    Tensor    dst0, dst0_f16;
    size_t    M             = 4;
    size_t    N             = 4;
    size_t    K             = 4;
    bool      default_input = true;

    // Parse args
    if(argc < 3) /* case default matrix sizes */
    {
        // Print help
        std::cout << "Usage: ./build/neon_gemm_qasymm8 M N K\n";
        std::cout << "Too few or no inputs provided. Using default M=4, N=4, K=4\n\n";
    }
    else /* case M N K arguments provided */
    {
        M             = strtol(argv[1], nullptr, 10);
        N             = strtol(argv[2], nullptr, 10);
        K             = strtol(argv[3], nullptr, 10);
        default_input = false;
    }

    /*** Floating point matrix multiplication ***/

    float * src1_ptr;
    float * src2_ptr;
    float * dst0_ptr;

    std::cout << " === f32 -> gemm -> f32 === " << std::endl;
    {
        // Initialise input matrices
        NEGEMM fgemm{};

        src1.allocator()->init(TensorInfo(TensorShape(K, M), 1, DataType::F32));
        src2.allocator()->init(TensorInfo(TensorShape(N, K), 1, DataType::F32));
        dst0.allocator()->init(TensorInfo(TensorShape(N, M), 1, DataType::F32));
        fgemm.configure(&src1, &src2, nullptr, &dst0, 1, 0);

        // Allocate matrices
        src1.allocator()->allocate();
        src2.allocator()->allocate();
        dst0.allocator()->allocate();

        // Fill in tensors, by default fill in with known data - for easy testing
        src1_ptr = reinterpret_cast<float *>(src1.buffer());
        src2_ptr = reinterpret_cast<float *>(src2.buffer());
        dst0_ptr = reinterpret_cast<float *>(dst0.buffer());

        // Fill in: one is the identity matrix, other is sequential values
        // src1: Identity matrix
        for(size_t i = 0; i < M * K; i++)
        {
            src1_ptr[i] = 0;
        }
        for(size_t i = 0; i < M; i++)
        {
            src1_ptr[i * K + i] = 1.0f;
        }

        // src2: Sequential values matrix
        for(size_t i = 0; i < K * N; i++)
        {
            src2_ptr[i] = i * 1.123f;
        }

        // Otherwise if M, N, K is given, fill in with random values
        if(!default_input)
        {
            fill_random_tensor(src1, 0.f, 1.f);
            fill_random_tensor(src2, 0.f, 1.f);
        }

        // Run single precision gemm and print result
        print_results(count_iter, [&]
                      { fgemm.run(); }, "fp32 time");
    }

    std::cout << " === f16 -> gemm -> f16 === " << std::endl;
    {
        NEGEMM fgemm_f16{};
        src1_f16.allocator()->init(TensorInfo(TensorShape(K, M), 1, DataType::F16));
        src2_f16.allocator()->init(TensorInfo(TensorShape(N, K), 1, DataType::F16));
        dst0_f16.allocator()->init(TensorInfo(TensorShape(N, M), 1, DataType::F16));
        fgemm_f16.configure(&src1_f16, &src2_f16, nullptr, &dst0_f16, 1, 0);

        // Allocate matrices
        src1_f16.allocator()->allocate();
        src2_f16.allocator()->allocate();
        dst0_f16.allocator()->allocate();

        // Fill in tensors, by default fill in with known data - for easy testing
        auto *src1_ptr_f16 = reinterpret_cast<float16_t *>(src1_f16.buffer());
        auto *src2_ptr_f16 = reinterpret_cast<float16_t *>(src2_f16.buffer());
        auto *dst0_ptr_f16 = reinterpret_cast<float16_t *>(dst0_f16.buffer());

        // Fill in: one is the identity matrix, other is sequential values
        // src1: Identity matrix
        for(size_t i = 0; i < M * K; i++)
        {
            src1_ptr_f16[i] = 0;
        }
        for(size_t i = 0; i < M; i++)
        {
            src1_ptr_f16[i * K + i] = 1.0f;
        }

        // src2: Sequential values matrix
        for(size_t i = 0; i < K * N; i++)
        {
            src2_ptr_f16[i] = i * 1.123f;
        }

        // Otherwise if M, N, K is given, fill in with random values
        if(!default_input)
        {
            fill_random_tensor(src1_f16, 0.f, 1.f);
            fill_random_tensor(src2_f16, 0.f, 1.f);
        }

        // Run single precision gemm f16 and print result
        print_results(count_iter, [&]
                      { fgemm_f16.run(); }, "fp16 time");
    }

    /*** Quantised asymmetric 8bit matrix  multiplication ***/

    // Start by finding the quantisation parameters for each set of values
    float src1_min;
    float src1_max;
    float src2_min;
    float src2_max;
    float dst0_min;
    float dst0_max;

    find_min_max(M * K, src1_ptr, &src1_min, &src1_max);
    find_min_max(K * N, src2_ptr, &src2_min, &src2_max);
    find_min_max(M * N, dst0_ptr, &dst0_min, &dst0_max);

    const QuantizationInfo src1_qinfo = choose_quantization_params(src1_min, src1_max);
    const QuantizationInfo src2_qinfo = choose_quantization_params(src2_min, src2_max);
    const QuantizationInfo dst0_qinfo = choose_quantization_params(dst0_min, dst0_max);

    //   std::cout << "Matrix 1: min=" << src1_min << ", max=" << src1_max << ", ";
    //   std::cout << "QuantisationInfo(" << src1_qinfo.scale()[0] << ", " << src1_qinfo.offset()[0] << ")\n";
    //   std::cout << "Matrix 2: min=" << src2_min << ", max=" << src2_max << ", ";
    //   std::cout << "QuantisationInfo(" << src2_qinfo.scale()[0] << ", " << src2_qinfo.offset()[0] << ")\n";
    //   std::cout << "Result  : min=" << dst0_min << ", max=" << dst0_max << ", ";
    //   std::cout << "QuantisationInfo(" << dst0_qinfo.scale()[0] << ", " << dst0_qinfo.offset()[0] << ")\n";
    std::vector<int *> dst32_ptr(2);
    std::vector<int8_t *> dst8_ptr(2);

    std::cout << " === gemm(s8, s8) -> s32 -> gemmlowp_output_stage -> s8 === " << std::endl;
    {
        Tensor q_src1;
        Tensor q_src2;
        Tensor q_dst0;
        Tensor q_res;
        Tensor q_res_output;
        // We now have the quantisation info and can configure the quantised tensors
        q_src1.allocator()->init(TensorInfo(TensorShape(K, M), 1, DataType::QASYMM8, src1_qinfo));
        q_src2.allocator()->init(TensorInfo(TensorShape(N, K), 1, DataType::QASYMM8, src2_qinfo));
        q_dst0.allocator()->init(TensorInfo(TensorShape(N, M), 1, DataType::QASYMM8, dst0_qinfo));

        // In this approach we use the QuantizationLayer construct to perform quantization
        NEQuantizationLayer q1;
        NEQuantizationLayer q2;
        NEQuantizationLayer q3;
        q1.configure(&src1, &q_src1);
        q2.configure(&src2, &q_src2);
        q3.configure(&dst0, &q_dst0);

        // Configure low precision gemm and initialise result tensor (pre-output)
        NEGEMMLowpMatrixMultiplyCore qgemm;
        q_res.allocator()->init(TensorInfo(TensorShape(N, M), 1, DataType::S32));
        qgemm.configure(&q_src1, &q_src2, nullptr, &q_res);

        // Configure output stage after computing shift and multiplier parameters
        NEGEMMLowpOutputStage gemmlowp_output_stage;
        int                   output_multiplier;
        int                   output_shift;
        float                 multiplier = (src1_qinfo.uniform().scale * src2_qinfo.uniform().scale) / dst0_qinfo.uniform().scale;
        quantization::calculate_quantized_multiplier_less_than_one(multiplier, &output_multiplier, &output_shift);
        //   std::cout << "(q_multiplier, q_shift) = (" << output_multiplier << ", " << output_shift << ")\n\n";

        GEMMLowpOutputStageInfo info;
        info.type                = GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT;
        info.gemmlowp_multiplier = output_multiplier;
        info.gemmlowp_shift      = output_shift;
        info.gemmlowp_offset     = dst0_qinfo.uniform().offset;
        info.output_data_type    = DataType::QASYMM8;
        q_res_output.info()->set_data_type(DataType::QASYMM8);
        q_res_output.info()->set_num_channels(1);
        gemmlowp_output_stage.configure(&q_res, nullptr, &q_res_output, info);

        // Allocate all tensors
        q_src1.allocator()->allocate();
        q_src2.allocator()->allocate();
        q_dst0.allocator()->allocate();
        q_res.allocator()->allocate();
        q_res_output.allocator()->allocate();

        // Run quantization layers (quantizes values of each tensor)
        q1.run();
        q2.run();
        q3.run();
        // Run low precision matrix multiply kernel
        print_results(count_iter, [&]
                      { qgemm.run(); }, "int8 time");

        // Run output stage kernel
        gemmlowp_output_stage.run();
        dst32_ptr[0] = reinterpret_cast<int *>(q_res.buffer());
        dst8_ptr[0] = reinterpret_cast<int8_t *>(q_res_output.buffer());
    }

    std::cout << " === gemm(i8(QI empty), i8(QI empty)) -> s32 -> gemmlowp_output_stage -> s8 === " << std::endl;
    {
        Tensor q_src1;
        Tensor q_src2;
        Tensor q_dst0;
        Tensor q_res;
        Tensor q_res_output;

        Tensor q_src1_tmp;
        Tensor q_src2_tmp;
        q_src1_tmp.allocator()->init(TensorInfo(TensorShape(K, M), 1, DataType::QASYMM8, src1_qinfo));
        q_src2_tmp.allocator()->init(TensorInfo(TensorShape(N, K), 1, DataType::QASYMM8, src2_qinfo));
        NEQuantizationLayer q1_tmp;
        NEQuantizationLayer q2_tmp;
        q1_tmp.configure(&src1, &q_src1_tmp);
        q2_tmp.configure(&src2, &q_src2_tmp);
        q_src1_tmp.allocator()->allocate();
        q_src2_tmp.allocator()->allocate();
        q1_tmp.run();
        q2_tmp.run();

        // We now have the quantisation info and can configure the quantised tensors
        q_src1.allocator()->init(TensorInfo(TensorShape(K, M), 1, DataType::QASYMM8, QuantizationInfo()));
        q_src2.allocator()->init(TensorInfo(TensorShape(N, K), 1, DataType::QASYMM8, QuantizationInfo()));
        q_dst0.allocator()->init(TensorInfo(TensorShape(N, M), 1, DataType::QASYMM8, dst0_qinfo));

        // In this approach we use the QuantizationLayer construct to perform quantization
        NEQuantizationLayer q3;
        q3.configure(&dst0, &q_dst0);

        // Configure low precision gemm and initialise result tensor (pre-output)
        NEGEMMLowpMatrixMultiplyCore qgemm;
        q_res.allocator()->init(TensorInfo(TensorShape(N, M), 1, DataType::S32));
        qgemm.configure(&q_src1, &q_src2, nullptr, &q_res);

        // Configure output stage after computing shift and multiplier parameters
        NEGEMMLowpOutputStage gemmlowp_output_stage;
        int                   output_multiplier;
        int                   output_shift;
        float                 multiplier = (src1_qinfo.uniform().scale * src2_qinfo.uniform().scale) / dst0_qinfo.uniform().scale;
        quantization::calculate_quantized_multiplier_less_than_one(multiplier, &output_multiplier, &output_shift);
        //   std::cout << "(q_multiplier, q_shift) = (" << output_multiplier << ", " << output_shift << ")\n\n";

        GEMMLowpOutputStageInfo info;
        info.type                = GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT;
        info.gemmlowp_multiplier = output_multiplier;
        info.gemmlowp_shift      = output_shift;
        info.gemmlowp_offset     = dst0_qinfo.uniform().offset;
        info.output_data_type    = DataType::QASYMM8;
        q_res_output.info()->set_data_type(DataType::QASYMM8);
        q_res_output.info()->set_num_channels(1);
        gemmlowp_output_stage.configure(&q_res, nullptr, &q_res_output, info);

        // Allocate all tensors
        q_src1.allocator()->import_memory(q_src1_tmp.allocator()->data());
        q_src2.allocator()->import_memory(q_src2_tmp.allocator()->data());
        q_dst0.allocator()->allocate();
        q_res.allocator()->allocate();
        q_res_output.allocator()->allocate();

        // Run quantization layers (quantizes values of each tensor)
        q3.run();
        // Run low precision matrix multiply kernel
        print_results(count_iter, [&]
                      { qgemm.run(); }, "int8 time");

        // Run output stage kernel
        gemmlowp_output_stage.run();
        dst32_ptr[1] = reinterpret_cast<int *>(q_res.buffer());
    }

    std::cout << " === gemm(s8, s8) -> (requant) -> s8 === " << std::endl;
    {
        Tensor q_src1;
        Tensor q_src2;
        Tensor q_dst0;
        // We now have the quantisation info and can configure the quantised tensors
        q_src1.allocator()->init(TensorInfo(TensorShape(K, M), 1, DataType::QASYMM8, src1_qinfo));
        q_src2.allocator()->init(TensorInfo(TensorShape(N, K), 1, DataType::QASYMM8, src2_qinfo));
        q_dst0.allocator()->init(TensorInfo(TensorShape(N, M), 1, DataType::QASYMM8, dst0_qinfo));

        // In this approach we use the QuantizationLayer construct to perform quantization
        NEQuantizationLayer q1;
        NEQuantizationLayer q2;
        NEQuantizationLayer q3;
        q1.configure(&src1, &q_src1);
        q2.configure(&src2, &q_src2);
        q3.configure(&dst0, &q_dst0);

        // Configure low precision gemm and initialise result tensor (pre-output)
        NEGEMMLowpMatrixMultiplyCore qgemm;
        qgemm.configure(&q_src1, &q_src2, nullptr, &q_dst0);

        // Allocate all tensors
        q_src1.allocator()->allocate();
        q_src2.allocator()->allocate();
        q_dst0.allocator()->allocate();

        // Run quantization layers (quantizes values of each tensor)
        q1.run();
        q2.run();
        q3.run();
        // Run low precision matrix multiply kernel
        print_results(count_iter, [&]
                      { qgemm.run(); }, "int8 time");

        dst8_ptr[1] = reinterpret_cast<int8_t *>(q_dst0.buffer());
    }

    for(int i = 0; i < N * M; i++) {
        if (dst32_ptr[0][i] != dst32_ptr[1][i]) {
            std::cout << "\nTest Failed\n";
            std::cout << "dst32_ptr[0]["<< i << "] = " << dst32_ptr[0][i] << "\n";
            std::cout << "dst32_ptr[1]["<< i << "] = " << dst32_ptr[1][i] << "\n";
            return -1;
        }
    }
    std::cout << "\nEmpty QuantizationInfo Test Passed\n";

    for(int i = 0; i < N * M; i++) {
        if (dst8_ptr[0][i] != dst8_ptr[1][i]) {
            std::cout << "\nTest Failed\n";
            std::cout << "dst8_ptr[0]["<< i << "] = " << static_cast<int>(dst8_ptr[0][i]) << "\n";
            std::cout << "dst8_ptr[1]["<< i << "] = " << static_cast<int>(dst8_ptr[1][i]) << "\n";
            return -1;
        }
    }
    std::cout << "\nInt8 requantization Test Passed\n";
}
