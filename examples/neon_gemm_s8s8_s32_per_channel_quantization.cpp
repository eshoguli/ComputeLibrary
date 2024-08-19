// scons arch=arm64-v8.2-a neon=1 opencl=0 openmp=0 cppthreads=1 os=macos data_layout_support=all  build=native asserts=1 debug=1 --jobs=8 --silent os=macos build=native fixed_format_kernels=True validation_tests=1 examples=1

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

std::vector<float> generate_quantization_scales(size_t channels, float start_value) {
    std::vector<float> values(channels);
    for(size_t i = 0; i < channels; i++) {
        values[i] = start_value + static_cast<float>(i) / 10.f;
    }
    return values;
}

std::ostream& operator<<(std::ostream& os, const TensorShape& shape) {
    return os << "[" << shape[0] << ", " << shape[1] << ", " << shape[2] << ", " << shape[3] << "]";
}

std::ostream& operator<<(std::ostream& os, const std::vector<float>& values) {
    os << "{";
    for(size_t i = 0; i < values.size(); i++) {
        std::cout << values[i] << " ";
    }
    os << "}";
    return os;
}

std::ostream& operator<<(std::ostream& os, const ITensorInfo* tensor_info) {
    const auto data_type = tensor_info->data_type();
    os << "pr=";
    switch (data_type) {
        case DataType::S8: {
            os << "S8";
            break;
        }
        case DataType::QSYMM8: {
            os << "QSYMM8";
            break;
        }
        case DataType::QASYMM8: {
            os << "QASYMM8";
            break;
        }
        case DataType::QASYMM8_SIGNED: {
            os << "QASYMM8_SIGNED";
            break;
        }
        case DataType::S32: {
            os << "S32";
            break;
        }
        case DataType::F32: {
            os << "F32";
            break;
        }
        default: {
            os << "[UNKNOWN]";
            break;
        }
    }

    const auto scales = tensor_info->quantization_info().scale();
    return os << " shape=" << tensor_info->tensor_shape() << " q=" << scales;
}

int main(int/* argc*/, char** /*argv*/)
{
    Tensor src1;
    Tensor src2;
    Tensor dst0;
    Tensor q_src1;
    Tensor q_src2;
    Tensor q_dst0;
    Tensor q_res;
    Tensor q_res_output;

    size_t n = 1;
    size_t c = 1;
    // A matrix: a1 x a2
    size_t a1 = 16;
    size_t a2 = 6;
    // B matrix: b1 x b2
    size_t b1 = 6;
    size_t b2 = 16;

    // Initialise input matrices
    src1.allocator()->init(TensorInfo(TensorShape(a2, a1, c, n), 1, DataType::F32));
    src2.allocator()->init(TensorInfo(TensorShape(b2, b1, c, n), 1, DataType::F32));

    // Allocate matrices
    src1.allocator()->allocate();
    src2.allocator()->allocate();

    // Fill in tensors, by default fill in with known data - for easy testing
    auto *src1_ptr = reinterpret_cast<float *>(src1.buffer());
    auto *src2_ptr = reinterpret_cast<float *>(src2.buffer());

    // Fill in: one is the identity matrix, other is sequential values
    // src1: Identity matrix
    for(size_t i_n = 0; i_n < n; i_n++) {
        for(size_t i_c = 0; i_c < c; i_c++) {
            for(size_t i_hw = 0; i_hw < a1 * a2; i_hw++)
            {
                //src1_ptr[i_hw + i_c * a1 * a2 + i_n * i_c * a1 * a2] = i_hw + i_c * a1 * a2 + i_n * i_c * a1 * a2;
                src1_ptr[i_hw + i_c * a1 * a2 + i_n * i_c * a1 * a2] = 1.f + static_cast<float>(i_c);
            }
        }
    }

    for(size_t i_n = 0; i_n < n; i_n++) {
        for(size_t i_c = 0; i_c < c; i_c++) {
            for(size_t i_hw = 0; i_hw < b1 * b2; i_hw++)
            {
                //src2_ptr[i_hw + i_c * b1 * b2 + i_n * i_c * b1 * b2] = i_hw + i_c * a1 * a2 + i_n * i_c * b1 * b2;
                src2_ptr[i_hw + i_c * b1 * b2 + i_n * i_c * b1 * b2] = -2.f - static_cast<float>(i_c);
            }
        }
    }

    std::cout << "src1 " << src1.info() << ":" << std::endl;
    src1.print(std::cout);
    std::cout << "src2 " << src2.info() << ":" << std::endl;
    src2.print(std::cout);

    const QuantizationInfo src1_qinfo(0.2f);
    
    // per-tensor quantization:
    //const QuantizationInfo src2_qinfo(0.2f);

    // per-channel quantization:
    const auto scales2 = generate_quantization_scales(6, 0.2f);
    const QuantizationInfo src2_qinfo(scales2);
    std::cout << "scales2: " << scales2 << std::endl;

    // We now have the quantisation info and can configure the quantised tensors
    q_src1.allocator()->init(TensorInfo(TensorShape(a2, a1, c, n), 1, DataType::QASYMM8_SIGNED, src1_qinfo));
    q_src2.allocator()->init(TensorInfo(TensorShape(b2, b1, c, n), 1, DataType::QASYMM8_SIGNED, src2_qinfo));

    // In this approach we use the QuantizationLayer construct to perform quantization
    NEQuantizationLayer q1;
    NEQuantizationLayer q2;
    q1.configure(&src1, &q_src1);
    q2.configure(&src2, &q_src2);

    // Configure low precision gemm and initialise result tensor (pre-output)
    NEGEMMLowpMatrixMultiplyCore qgemm;
    q_res.allocator()->init(TensorInfo(TensorShape(a1, b2, c, n), 1, DataType::S32));
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

//#if ARM_COMPUTE_DEBUG_ENABLED
    // Print quantized source matrices
    std::cout << "q_src1 " << q_src1.info() << ":" << std::endl;
    q_src1.print(std::cout);
    std::cout << "q_src2 " << q_src2.info() << ":" << std::endl;
    q_src2.print(std::cout);
    // Print result matrix in int32 form - before output stage processing
    std::cout << "Lowp GEMM output " << q_res.info() << ":" << std::endl;
    q_res.print(std::cout);
//#endif // ARM_COMPUTE_DEBUG_ENABLED

    return 0;
}
