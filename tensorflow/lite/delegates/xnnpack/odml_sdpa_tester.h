/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_ODML_SDPA_TESTER_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_ODML_SDPA_TESTER_H_

#include <cstdint>
#include <functional>
#include <random>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/builtin_op_data.h"

namespace tflite {
namespace xnnpack {

class ODMLSDPATester {
 public:
  ODMLSDPATester() = default;
  ODMLSDPATester(const ODMLSDPATester&) = delete;
  ODMLSDPATester& operator=(const ODMLSDPATester&) = delete;

  inline ODMLSDPATester& Input1Shape(std::initializer_list<int32_t> shape) {
    EXPECT_THAT(shape, testing::Each(testing::Gt(0)));
    input1_shape_ = std::vector<int32_t>(shape.begin(), shape.end());
    input1_size_ = ComputeSize(input1_shape_);
    return *this;
  }

  inline const std::vector<int32_t>& Input1Shape() const {
    return input1_shape_;
  }

  inline ODMLSDPATester& Input2Shape(std::initializer_list<int32_t> shape) {
    EXPECT_THAT(shape, testing::Each(testing::Gt(0)));
    input2_shape_ = std::vector<int32_t>(shape.begin(), shape.end());
    input2_size_ = ComputeSize(input2_shape_);
    return *this;
  }

  inline const std::vector<int32_t>& Input2Shape() const {
    return input2_shape_;
  }

  inline ODMLSDPATester& Input3Shape(std::initializer_list<int32_t> shape) {
    EXPECT_THAT(shape, testing::Each(testing::Gt(0)));
    input3_shape_ = std::vector<int32_t>(shape.begin(), shape.end());
    input3_size_ = ComputeSize(input3_shape_);
    return *this;
  }

  inline const std::vector<int32_t>& Input3Shape() const {
    return input3_shape_;
  }

  inline ODMLSDPATester& Input4Shape(std::initializer_list<int32_t> shape) {
    EXPECT_THAT(shape, testing::Each(testing::Gt(0)));
    input4_shape_ = std::vector<int32_t>(shape.begin(), shape.end());
    input4_size_ = ComputeSize(input4_shape_);
    return *this;
  }

  inline const std::vector<int32_t>& Input4Shape() const {
    return input4_shape_;
  }

  inline ODMLSDPATester& Input5Shape(std::initializer_list<int32_t> shape) {
    EXPECT_THAT(shape, testing::Each(testing::Gt(0)));
    input5_shape_ = std::vector<int32_t>(shape.begin(), shape.end());
    input5_size_ = ComputeSize(input5_shape_);
    return *this;
  }

  inline const std::vector<int32_t>& Input5Shape() const {
    return input5_shape_;
  }

  inline int32_t Input1Size() const { return input1_size_; }

  inline int32_t Input2Size() const { return input2_size_; }

  inline int32_t Input3Size() const { return input3_size_; }

  inline int32_t Input4Size() const { return input4_size_; }

  inline int32_t Input5Size() const { return input5_size_; }

  std::vector<int32_t> OutputShape() const;

  static int32_t ComputeSize(const std::vector<int32_t>& shape);

  void Test(TfLiteDelegate* delegate) const;

 private:
  std::vector<char> CreateTfLiteModel() const;

  std::vector<int32_t> input1_shape_;
  std::vector<int32_t> input2_shape_;
  std::vector<int32_t> input3_shape_;
  std::vector<int32_t> input4_shape_;
  std::vector<int32_t> input5_shape_;
  int32_t input1_size_ = 1;
  int32_t input2_size_ = 1;
  int32_t input3_size_ = 1;
  int32_t input4_size_ = 1;
  int32_t input5_size_ = 1;
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_ODML_SDPA_TESTER_H_
