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
#include "arm_compute/core/WindowIterator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "support/ToolchainSupport.h"
#include "utils/Utils.h"

#include <cstdlib>

using namespace arm_compute;
using namespace utils;

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
    if(zero_point_real < qmin)
    {
        zero_point_nudged = qmin;
    }
    else if(zero_point_real > qmax)
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
    for(int i = 0; i < size; i++)
    {
        output[i] = quantize_qasymm8(input[i], qinfo);
    }
    std::cout << "\n";
}

std::ostream& operator<<(std::ostream& os, const ITensorInfo* tensor_info) {
    const auto data_type = tensor_info->data_type();
    switch (data_type) {
        case DataType::S8: {
            return os << "S8";
        }
        case DataType::QSYMM8: {
            return os << "QSYMM8";
        }
        case DataType::QASYMM8: {
            return os << "QASYMM8";
        }
        case DataType::QASYMM8_SIGNED: {
            return os << "QASYMM8_SIGNED";
        }
        case DataType::S32: {
            return os << "S32";
        }
        case DataType::F32: {
            return os << "F32";
        }
        default: {
            return os << "[UNKNOWN]";
        }
    }
}

int main(int argc, char **argv)
{
    Tensor src1;
    Tensor src2;
    Tensor dst0;
    Tensor q_src1;
    Tensor q_src2;
    Tensor q_dst0;
    Tensor q_res;
    Tensor q_res_output;
    size_t M             = 4;
    size_t N             = 4;
    size_t K             = 4;
    bool   default_input = true;

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

    // Initialise input matrices
    src1.allocator()->init(TensorInfo(TensorShape(K, M), 1, DataType::F32));
    src2.allocator()->init(TensorInfo(TensorShape(N, K), 1, DataType::F32));

    // Allocate matrices
    src1.allocator()->allocate();
    src2.allocator()->allocate();
    //dst0.allocator()->allocate();

    // Fill in tensors, by default fill in with known data - for easy testing
    auto *src1_ptr = reinterpret_cast<float *>(src1.buffer());
    auto *src2_ptr = reinterpret_cast<float *>(src2.buffer());

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
        src2_ptr[i] = i * (i % 2 ? 1.123f : -1.123f);
    }

    // Otherwise if M, N, K is given, fill in with random values
    if(!default_input)
    {
        fill_random_tensor(src1, 0.f, 1.f);
        fill_random_tensor(src2, 0.f, 1.f);
    }

    // original
    // const QuantizationInfo src1_qinfo(0.00392157f);
    // const QuantizationInfo src2_qinfo(0.0660588f);

    const QuantizationInfo src1_qinfo(0.0392157f);
    const QuantizationInfo src2_qinfo(0.660588f);

    // We now have the quantisation info and can configure the quantised tensors
    q_src1.allocator()->init(TensorInfo(TensorShape(K, M), 1, DataType::QASYMM8, src1_qinfo));
    q_src2.allocator()->init(TensorInfo(TensorShape(N, K), 1, DataType::QASYMM8, src2_qinfo));

    // In this approach we use the QuantizationLayer construct to perform quantization
    NEQuantizationLayer q1;
    NEQuantizationLayer q2;
    q1.configure(&src1, &q_src1);
    q2.configure(&src2, &q_src2);

    // Configure low precision gemm and initialise result tensor (pre-output)
    NEGEMMLowpMatrixMultiplyCore qgemm;
    q_res.allocator()->init(TensorInfo(TensorShape(N, M), 1, DataType::S32));
    qgemm.configure(&q_src1, &q_src2, nullptr, &q_res);

    // Allocate all tensors
    q_src1.allocator()->allocate();
    q_src2.allocator()->allocate();
    q_dst0.allocator()->allocate();
    q_res.allocator()->allocate();
    q_res_output.allocator()->allocate();

    // Run quantization layers (quantizes values of each tensor)
    q1.run();
    q2.run();
    // Run low precision matrix multiply kernel
    qgemm.run();

#if ARM_COMPUTE_DEBUG_ENABLED
    // Print quantized source matrices
    std::cout << "q_src1 " << q_src1.info() << ":" << std::endl;
    q_src1.print(std::cout);
    std::cout << "q_src2 " << q_src2.info() << ":" << std::endl;
    q_src2.print(std::cout);
    // Print result matrix in int32 form - before output stage processing
    std::cout << "Lowp GEMM output " << q_res.info() << ":" << std::endl;
    q_res.print(std::cout);
#endif // ARM_COMPUTE_DEBUG_ENABLED
}
