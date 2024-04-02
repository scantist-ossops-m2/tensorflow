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

#include "tensorflow/lite/delegates/xnnpack/odml_sdpa_tester.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "flatbuffers/string.h"  // from @flatbuffers
#include "tensorflow/lite/core/kernels/register.h"
#include "tensorflow/lite/experimental/genai/genai_ops.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/schema/schema_conversion_utils.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace tflite {
namespace xnnpack {

std::vector<int32_t> ODMLSDPATester::OutputShape() const {
  std::vector<int32_t> output_shape = Input1Shape();
  return output_shape;
}

void ODMLSDPATester::Test(TfLiteDelegate* delegate) const {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto input_rng =
      std::bind(std::uniform_real_distribution<float>(), std::ref(rng));

  std::vector<char> buffer = CreateTfLiteModel();
  const Model* model = GetModel(buffer.data());

  std::unique_ptr<Interpreter> delegate_interpreter;
  auto resolver =
      ::tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates();
  resolver.AddCustom("odml.scaled_dot_product_attention",
                     tflite::ops::custom::Register_SDPA());
  ASSERT_EQ(InterpreterBuilder(model, resolver)(&delegate_interpreter),
            kTfLiteOk);
  std::unique_ptr<Interpreter> default_interpreter;
  ASSERT_EQ(InterpreterBuilder(model, resolver)(&default_interpreter),
            kTfLiteOk);

  ASSERT_TRUE(delegate_interpreter);
  ASSERT_TRUE(default_interpreter);

  ASSERT_EQ(delegate_interpreter->inputs().size(), 5);
  ASSERT_EQ(default_interpreter->inputs().size(), 5);

  ASSERT_EQ(delegate_interpreter->outputs().size(), 1);
  ASSERT_EQ(default_interpreter->outputs().size(), 1);

  ASSERT_EQ(delegate_interpreter->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(default_interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(delegate_interpreter->ModifyGraphWithDelegate(delegate), kTfLiteOk);

  float* delegate_input1_data =
      delegate_interpreter->typed_input_tensor<float>(0);
  float* delegate_input2_data =
      delegate_interpreter->typed_input_tensor<float>(1);
  float* delegate_input3_data =
      delegate_interpreter->typed_input_tensor<float>(2);
  float* delegate_input4_data =
      delegate_interpreter->typed_input_tensor<float>(3);
  float* delegate_input5_data =
      delegate_interpreter->typed_input_tensor<float>(4);
  std::generate_n(delegate_input1_data, Input1Size(), std::ref(input_rng));
  std::generate_n(delegate_input2_data, Input2Size(), std::ref(input_rng));
  std::generate_n(delegate_input3_data, Input3Size(), std::ref(input_rng));
  std::generate_n(delegate_input4_data, Input4Size(), std::ref(input_rng));
  std::generate_n(delegate_input5_data, Input5Size(), std::ref(input_rng));

  float* default_input1_data =
      default_interpreter->typed_input_tensor<float>(0);
  float* default_input2_data =
      default_interpreter->typed_input_tensor<float>(1);
  float* default_input3_data =
      default_interpreter->typed_input_tensor<float>(2);
  float* default_input4_data =
      default_interpreter->typed_input_tensor<float>(3);
  float* default_input5_data =
      default_interpreter->typed_input_tensor<float>(4);
  std::copy_n(delegate_input1_data, Input1Size(), default_input1_data);
  std::copy_n(delegate_input2_data, Input2Size(), default_input2_data);
  std::copy_n(delegate_input3_data, Input3Size(), default_input3_data);
  std::copy_n(delegate_input4_data, Input4Size(), default_input4_data);
  std::copy_n(delegate_input5_data, Input5Size(), default_input5_data);

  ASSERT_EQ(delegate_interpreter->Invoke(), kTfLiteOk);
  ASSERT_EQ(default_interpreter->Invoke(), kTfLiteOk);

  float* delegate_output_data =
      delegate_interpreter->typed_output_tensor<float>(0);
  float* default_output_data =
      default_interpreter->typed_output_tensor<float>(0);
  const int32_t output_size = ComputeSize(OutputShape());

  for (size_t i = 0; i < output_size; i++) {
    ASSERT_NEAR(default_output_data[i], delegate_output_data[i],
                std::numeric_limits<float>::epsilon() *
                    std::max(std::abs(default_output_data[i]) * 20.0f, 1.0f));
  }
}

std::vector<char> ODMLSDPATester::CreateTfLiteModel() const {
  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<OperatorCode> operator_code = CreateOperatorCode(
      builder, BuiltinOperator_CUSTOM,
      builder.CreateString("odml.scaled_dot_product_attention"));

  const std::array<flatbuffers::Offset<Buffer>, 1> buffers{{
      CreateBuffer(builder, builder.CreateVector({})),
  }};

  const std::array<flatbuffers::Offset<Tensor>, 6> tensors{{
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(Input1Shape().data(),
                                                 Input1Shape().size()),
                   TensorType_FLOAT32),
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(Input2Shape().data(),
                                                 Input2Shape().size()),
                   TensorType_FLOAT32),
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(Input3Shape().data(),
                                                 Input3Shape().size()),
                   TensorType_FLOAT32),
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(Input4Shape().data(),
                                                 Input4Shape().size()),
                   TensorType_FLOAT32),
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(Input5Shape().data(),
                                                 Input5Shape().size()),
                   TensorType_FLOAT32),
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(OutputShape().data(),
                                                 OutputShape().size()),
                   TensorType_FLOAT32),
  }};

  const std::array<int32_t, 5> op_inputs{{0, 1, 2, 3, 4}};
  const std::array<int32_t, 1> op_outputs{{5}};
  flatbuffers::Offset<Operator> op = CreateOperator(
      builder, /*opcode_index=*/0,
      builder.CreateVector<int32_t>(op_inputs.data(), op_inputs.size()),
      builder.CreateVector<int32_t>(op_outputs.data(), op_outputs.size()),
      tflite::BuiltinOptions_NONE, 0);

  const std::array<int32_t, 5> subgraph_inputs{{0, 1, 2, 3, 4}};
  const std::array<int32_t, 1> subgraph_outputs{{5}};
  flatbuffers::Offset<SubGraph> subgraph = CreateSubGraph(
      builder, builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>(subgraph_inputs.data(),
                                    subgraph_inputs.size()),
      builder.CreateVector<int32_t>(subgraph_outputs.data(),
                                    subgraph_outputs.size()),
      builder.CreateVector(&op, 1));

  flatbuffers::Offset<flatbuffers::String> description =
      builder.CreateString("ODML SDPA model");

  flatbuffers::Offset<Model> model_buffer = CreateModel(
      builder, TFLITE_SCHEMA_VERSION, builder.CreateVector(&operator_code, 1),
      builder.CreateVector(&subgraph, 1), description,
      builder.CreateVector(buffers.data(), buffers.size()));

  builder.Finish(model_buffer);

  return std::vector<char>(builder.GetBufferPointer(),
                           builder.GetBufferPointer() + builder.GetSize());
}

int32_t ODMLSDPATester::ComputeSize(const std::vector<int32_t>& shape) {
  return std::accumulate(shape.cbegin(), shape.cend(), 1,
                         std::multiplies<int32_t>());
}

}  // namespace xnnpack
}  // namespace tflite
