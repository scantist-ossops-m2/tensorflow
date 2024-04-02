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

#include <math.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <vector>

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/reference/add.h"
#include "tensorflow/lite/kernels/internal/reference/batch_matmul.h"
#include "tensorflow/lite/kernels/internal/reference/fully_connected.h"
#include "tensorflow/lite/kernels/internal/reference/softmax.h"
#include "tensorflow/lite/kernels/internal/reference/transpose.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace llm {

static const int kQueryTensor = 0;
static const int kKeyTensor = 1;
static const int kValueTensor = 2;
static const int kAttentionMaskTensor = 3;
static const int kScaleTensor = 4;
static const int kOutputTensor = 0;

struct OpData {
  float scale;
  int scratch_tensor_index;
};

void* SDPAInit(TfLiteContext* context, const char* buffer, size_t length) {
  OpData* op_data = new OpData();
  op_data->scale = 0.0f;
  context->AddTensors(context, 8, &op_data->scratch_tensor_index);
  return op_data;
}

static std::vector<size_t> GetOutDimsForElementWise(int* lhs_dims,
                                                    const int* rhs_dims,
                                                    int num_dims) {
  std::vector<size_t> lhs_dim;
  lhs_dim.reserve(num_dims);
  for (int i = 0; i < num_dims; ++i) lhs_dim.push_back(lhs_dims[i]);
  std::vector<size_t> rhs_dim;
  rhs_dim.reserve(num_dims);
  for (int i = 0; i < num_dims; ++i) rhs_dim.push_back(rhs_dims[i]);
  TFLITE_DCHECK(!lhs_dim.empty());
  TFLITE_DCHECK(!rhs_dim.empty());
  std::vector<size_t> lhs_dims_rev(lhs_dim.rbegin(), lhs_dim.rend());
  std::vector<size_t> rhs_dims_rev(rhs_dim.rbegin(), rhs_dim.rend());
  TFLITE_DCHECK([&]() -> bool {
    for (size_t i = 0; i < std::min(lhs_dims_rev.size(), rhs_dims_rev.size());
         ++i) {
      if ((lhs_dims_rev[i] != rhs_dims_rev[i]) && (lhs_dims_rev[i] != 1) &&
          (rhs_dims_rev[i] != 1)) {
        return false;
      }
    }
    return true;
  }());
  std::vector<size_t> out_dims(
      std::max(lhs_dims_rev.size(), rhs_dims_rev.size()));
  for (int i = 0; i < out_dims.size(); ++i) {
    if (lhs_dims_rev.size() <= i) {
      out_dims[i] = rhs_dims_rev[i];
    } else if (rhs_dims_rev.size() <= i) {
      out_dims[i] = lhs_dims_rev[i];
    } else {
      out_dims[i] = lhs_dims_rev[i] == 1 ? rhs_dims_rev[i] : lhs_dims_rev[i];
    }
  }
  return std::vector<size_t>(out_dims.rbegin(), out_dims.rend());
}

TfLiteStatus SDPAPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 5);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);

  const TfLiteTensor* q_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kQueryTensor, &q_tensor));
  const TfLiteTensor* k_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kKeyTensor, &k_tensor));
  const TfLiteTensor* v_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kValueTensor, &v_tensor));
  const TfLiteTensor* mask_tensor;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kAttentionMaskTensor, &mask_tensor));
  const TfLiteTensor* scale_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kScaleTensor, &scale_tensor));
  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = TfLiteIntArrayCreate(8);
  bool mqa = k_tensor->dims->data[2] == 1;

  // Temp tensor for Transposed Q;
  {
    node->temporaries->data[0] = op_data->scratch_tensor_index;
    TfLiteTensor* scratch_buffer;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, /*index=*/0, &scratch_buffer));
    TfLiteIntArray* scratch_buffer_size = TfLiteIntArrayCreate(4);
    for (int i = 0; i < 4; ++i) {
      scratch_buffer_size->data[i] = q_tensor->dims->data[i];
    }
    // Swap middle two dimensions.
    scratch_buffer_size->data[1] = q_tensor->dims->data[2];
    scratch_buffer_size->data[2] = q_tensor->dims->data[1];

    scratch_buffer->type = kTfLiteFloat32;
    scratch_buffer->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scratch_buffer,
                                                     scratch_buffer_size));
  }

  // Temp tensor for Transposed K;
  {
    node->temporaries->data[1] = op_data->scratch_tensor_index + 1;
    TfLiteTensor* scratch_buffer;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, /*index=*/1, &scratch_buffer));
    TfLiteIntArray* scratch_buffer_size = TfLiteIntArrayCreate(4);
    for (int i = 0; i < 4; ++i) {
      scratch_buffer_size->data[i] = k_tensor->dims->data[i];
    }
    // Swap to middle two dimensions.
    scratch_buffer_size->data[1] = k_tensor->dims->data[2];
    scratch_buffer_size->data[2] = k_tensor->dims->data[1];

    scratch_buffer->type = kTfLiteFloat32;
    scratch_buffer->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scratch_buffer,
                                                     scratch_buffer_size));
  }

  // Temp tensor for Matmul1 output;
  int matmul1_out_shape[4];
  {
    node->temporaries->data[2] = op_data->scratch_tensor_index + 2;
    TfLiteTensor* scratch_buffer;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, /*index=*/2, &scratch_buffer));
    TfLiteIntArray* scratch_buffer_size = TfLiteIntArrayCreate(4);
    // mha/gqa: [permute_q[0], permute_q[1], permute_q[2], permute_k[2]]
    int matmul_out_shape[4] = {q_tensor->dims->data[0], q_tensor->dims->data[2],
                               q_tensor->dims->data[1],
                               k_tensor->dims->data[1]};
    for (int i = 0; i < 4; ++i) {
      scratch_buffer_size->data[i] = matmul_out_shape[i];
      matmul1_out_shape[i] = matmul_out_shape[i];
    }

    scratch_buffer->type = kTfLiteFloat32;
    scratch_buffer->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scratch_buffer,
                                                     scratch_buffer_size));
  }

  // Temp tensor for add output;
  int add_out_shape[4];
  {
    node->temporaries->data[3] = op_data->scratch_tensor_index + 3;
    TfLiteTensor* scratch_buffer;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, /*index=*/3, &scratch_buffer));
    TfLiteIntArray* scratch_buffer_size = TfLiteIntArrayCreate(4);
    // get dims from attention_mask, matmul1_out
    auto add_out_dims =
        GetOutDimsForElementWise(mask_tensor->dims->data, matmul1_out_shape, 4);
    for (int i = 0; i < 4; ++i) {
      scratch_buffer_size->data[i] = add_out_dims[i];
      add_out_shape[i] = add_out_dims[i];
    }

    scratch_buffer->type = kTfLiteFloat32;
    scratch_buffer->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scratch_buffer,
                                                     scratch_buffer_size));
  }

  // Temp tensor for Transposed V;
  {
    node->temporaries->data[4] = op_data->scratch_tensor_index + 4;
    TfLiteTensor* scratch_buffer;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, /*index=*/4, &scratch_buffer));
    TfLiteIntArray* scratch_buffer_size = TfLiteIntArrayCreate(4);
    // Swap to {0, 2, 3, 1} dimensions.
    scratch_buffer_size->data[0] = v_tensor->dims->data[0];
    scratch_buffer_size->data[1] = v_tensor->dims->data[2];
    scratch_buffer_size->data[2] = v_tensor->dims->data[3];
    scratch_buffer_size->data[3] = v_tensor->dims->data[1];

    scratch_buffer->type = kTfLiteFloat32;
    scratch_buffer->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scratch_buffer,
                                                     scratch_buffer_size));
  }

  // Temp tensor for Matmul2 output;
  {
    node->temporaries->data[5] = op_data->scratch_tensor_index + 5;
    TfLiteTensor* scratch_buffer;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, /*index=*/5, &scratch_buffer));
    TfLiteIntArray* scratch_buffer_size = TfLiteIntArrayCreate(4);
    // logits_out_shape = add_out_shape
    // mha/gqa: [logits_out[0], logits_out[1], logits_out[2], permute_v[2]]
    scratch_buffer_size->data[0] = add_out_shape[0];
    scratch_buffer_size->data[1] = add_out_shape[1];
    scratch_buffer_size->data[2] = add_out_shape[2];
    scratch_buffer_size->data[3] = v_tensor->dims->data[3];

    scratch_buffer->type = kTfLiteFloat32;
    scratch_buffer->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scratch_buffer,
                                                     scratch_buffer_size));
  }

  // Temp tensor for Reshape K / Transpose Q;
  {
    node->temporaries->data[6] = op_data->scratch_tensor_index + 6;
    TfLiteTensor* scratch_buffer;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, /*index=*/6, &scratch_buffer));
    TfLiteIntArray* scratch_buffer_size;
    if (mqa)
      scratch_buffer_size = TfLiteIntArrayCreate(2);
    else
      scratch_buffer_size = TfLiteIntArrayCreate(4);
    if (mqa) {
      scratch_buffer_size->data[0] = k_tensor->dims->data[1];
      scratch_buffer_size->data[1] = k_tensor->dims->data[3];
    } else {
      scratch_buffer_size->data[0] = q_tensor->dims->data[0];
      scratch_buffer_size->data[1] = q_tensor->dims->data[2];
      scratch_buffer_size->data[2] = q_tensor->dims->data[3];
      scratch_buffer_size->data[3] = q_tensor->dims->data[1];
    }

    scratch_buffer->type = kTfLiteFloat32;
    scratch_buffer->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scratch_buffer,
                                                     scratch_buffer_size));
  }

  // Temp tensor for Reshape V / Add_out (softmax_out);
  {
    node->temporaries->data[7] = op_data->scratch_tensor_index + 7;
    TfLiteTensor* scratch_buffer;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, /*index=*/7, &scratch_buffer));
    TfLiteIntArray* scratch_buffer_size;
    if (mqa)
      scratch_buffer_size = TfLiteIntArrayCreate(2);
    else
      scratch_buffer_size = TfLiteIntArrayCreate(4);
    if (mqa) {
      scratch_buffer_size->data[0] = v_tensor->dims->data[3];
      scratch_buffer_size->data[1] = v_tensor->dims->data[1];
    } else {
      scratch_buffer_size->data[0] = add_out_shape[0];
      scratch_buffer_size->data[1] = add_out_shape[1];
      scratch_buffer_size->data[2] = add_out_shape[3];
      scratch_buffer_size->data[3] = add_out_shape[2];
    }

    scratch_buffer->type = kTfLiteFloat32;
    scratch_buffer->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scratch_buffer,
                                                     scratch_buffer_size));
  }

  return kTfLiteOk;
}

void SDPAFree(TfLiteContext* context, void* buffer) {
  delete static_cast<OpData*>(buffer);
}

TfLiteStatus SDPAEval(TfLiteContext* context, TfLiteNode* node) {
  /*
  Simple implementation of Scaled Dot Product Attention.
  Takes query_proj, key_proj, value_proj, scale, mask tensors as inputs, and
  outputs the attention result.

  Notes:
  Scale is computed using 1/sqrt(head_dim),
  head_dim = q[-1] = embedding_dim // num_q_heads
  Only support for FLOAT32 inputs for now.
  Only support static tensors for now (k/v[1] = max sequence length)
  */

  const TfLiteTensor* query_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kQueryTensor, &query_tensor));
  auto query_shape = GetTensorShape(query_tensor);
  auto query_data = GetTensorData<float>(query_tensor);
  const TfLiteTensor* key_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kKeyTensor, &key_tensor));
  auto key_shape = GetTensorShape(key_tensor);
  auto key_data = GetTensorData<float>(key_tensor);
  const TfLiteTensor* value_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kValueTensor, &value_tensor));
  auto value_shape = GetTensorShape(value_tensor);
  auto value_data = GetTensorData<float>(value_tensor);
  const TfLiteTensor* attention_mask_tensor;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kAttentionMaskTensor,
                                          &attention_mask_tensor));
  auto attention_mask_shape = GetTensorShape(attention_mask_tensor);
  auto attention_mask_data = GetTensorData<float>(attention_mask_tensor);
  // not using scale tensor for input (use internal calculated or attr)
  TfLiteTensor* output_tensor;
  TF_LITE_ENSURE_OK(
      context, GetOutputSafe(context, node, kOutputTensor, &output_tensor));
  auto output_shape = GetTensorShape(output_tensor);
  auto output_data = GetTensorData<float>(output_tensor);

  // temporaries
  TfLiteTensor* transpose_q_out_tensor;
  TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/0,
                                              &transpose_q_out_tensor));
  auto transpose_q_out_shape = GetTensorShape(transpose_q_out_tensor);
  auto transpose_q_out_data = GetTensorData<float>(transpose_q_out_tensor);
  TfLiteTensor* transpose_k_out_tensor;
  TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/1,
                                              &transpose_k_out_tensor));
  auto transpose_k_out_shape = GetTensorShape(transpose_k_out_tensor);
  auto transpose_k_out_data = GetTensorData<float>(transpose_k_out_tensor);
  TfLiteTensor* matmul1_out_tensor;
  TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/2,
                                              &matmul1_out_tensor));
  auto matmul1_out_shape = GetTensorShape(matmul1_out_tensor);
  auto matmul1_out_data = GetTensorData<float>(matmul1_out_tensor);
  TfLiteTensor* add_out_tensor;
  TF_LITE_ENSURE_OK(
      context, GetTemporarySafe(context, node, /*index=*/3, &add_out_tensor));
  auto add_out_shape = GetTensorShape(add_out_tensor);
  auto add_out_data = GetTensorData<float>(add_out_tensor);
  TfLiteTensor* transpose_v_out_tensor;
  TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/4,
                                              &transpose_v_out_tensor));
  auto transpose_v_out_shape = GetTensorShape(transpose_v_out_tensor);
  auto transpose_v_out_data = GetTensorData<float>(transpose_v_out_tensor);
  TfLiteTensor* matmul2_out_tensor;
  TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/5,
                                              &matmul2_out_tensor));
  auto matmul2_out_shape = GetTensorShape(matmul2_out_tensor);
  auto matmul2_out_data = GetTensorData<float>(matmul2_out_tensor);
  TfLiteTensor* reshape_k_or_q_out_tensor;
  TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/6,
                                              &reshape_k_or_q_out_tensor));
  auto reshape_k_or_q_out_shape = GetTensorShape(reshape_k_or_q_out_tensor);
  auto reshape_k_or_q_out_data =
      GetTensorData<float>(reshape_k_or_q_out_tensor);
  TfLiteTensor* reshape_v_or_add_out_tensor;
  TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/7,
                                              &reshape_v_or_add_out_tensor));
  auto reshape_v_or_add_out_shape = GetTensorShape(reshape_v_or_add_out_tensor);
  auto reshape_v_or_add_out_data =
      GetTensorData<float>(reshape_v_or_add_out_tensor);

  bool mqa = key_tensor->dims->data[2] == 1;

  // TODO(b/329465380): get scale from attr
  // scale * q
  float scale = 1 / sqrt(query_tensor->dims->data[3]);
  int flat_size = query_shape.FlatSize();
  float output_min = -std::numeric_limits<float>::infinity();
  float output_max = std::numeric_limits<float>::infinity();
  for (int i = 0; i < flat_size; ++i) {
    query_tensor->data.f[i] = ActivationFunctionWithMinMax(
        query_tensor->data.f[i] * scale, output_min, output_max);
  }

  // permute q {0, 2, 1, 3}
  tflite::TransposeParams transpose_q_params;
  transpose_q_params.perm_count = 4;
  transpose_q_params.perm[0] = 0;
  transpose_q_params.perm[1] = 2;
  transpose_q_params.perm[2] = 1;
  transpose_q_params.perm[3] = 3;
  reference_ops::Transpose(transpose_q_params, query_shape, query_data,
                           transpose_q_out_shape, transpose_q_out_data);

  // permute k {0, 2, 1, 3}
  tflite::TransposeParams transpose_k_params;
  transpose_k_params.perm_count = 4;
  transpose_k_params.perm[0] = 0;
  transpose_k_params.perm[1] = 2;
  transpose_k_params.perm[2] = 1;
  transpose_k_params.perm[3] = 3;
  reference_ops::Transpose(transpose_k_params, key_shape, key_data,
                           transpose_k_out_shape, transpose_k_out_data);

  // reshape k for MQA, or transpose q for MHA
  if (mqa) {
    TF_LITE_ENSURE_EQ(context, transpose_k_out_tensor->bytes,
                      reshape_k_or_q_out_tensor->bytes);
    memcpy(reshape_k_or_q_out_tensor->data.data,
           transpose_k_out_tensor->data.data, transpose_k_out_tensor->bytes);
  } else {
    // permute q2 {0, 1, 3, 2}
    tflite::TransposeParams transpose_q2_params;
    transpose_q2_params.perm_count = 4;
    transpose_q2_params.perm[0] = 0;
    transpose_q2_params.perm[1] = 1;
    transpose_q2_params.perm[2] = 3;
    transpose_q2_params.perm[3] = 2;
    reference_ops::Transpose(transpose_q2_params, transpose_q_out_shape,
                             transpose_q_out_data, reshape_k_or_q_out_shape,
                             reshape_k_or_q_out_data);
  }

  // mqa FC (q, squeezed_k)
  // mha BMM(q, k) transpose_b = true
  if (mqa) {
    tflite::FullyConnectedParams fc_params;
    fc_params.float_activation_min = output_min;
    fc_params.float_activation_max = output_max;
    reference_ops::FullyConnected(
        fc_params, transpose_q_out_shape, transpose_q_out_data,
        reshape_k_or_q_out_shape, reshape_k_or_q_out_data, RuntimeShape(),
        nullptr, matmul1_out_shape, matmul1_out_data);
  } else {
    // pass rhs first (this is why we transpose q above)
    reference_ops::BatchMatMul(
        transpose_k_out_shape, transpose_k_out_data, reshape_k_or_q_out_shape,
        reshape_k_or_q_out_data, matmul1_out_shape, matmul1_out_data);
  }

  // add matmul_out + mask
  tflite::ArithmeticParams add_params;
  SetActivationParams(output_min, output_max, &add_params);
  reference_ops::BroadcastAdd6DSlow(
      add_params, attention_mask_shape, attention_mask_data, matmul1_out_shape,
      matmul1_out_data, add_out_shape, add_out_data);

  // softmax, can do in-place
  tflite::SoftmaxParams softmax_params;
  softmax_params.beta = 1.0f;
  reference_ops::Softmax(softmax_params, add_out_shape, add_out_data,
                         add_out_shape, add_out_data);

  // permute v {0, 2, 3, 1}
  tflite::TransposeParams transpose_v_params;
  transpose_v_params.perm_count = 4;
  transpose_v_params.perm[0] = 0;
  transpose_v_params.perm[1] = 2;
  transpose_v_params.perm[2] = 3;
  transpose_v_params.perm[3] = 1;
  reference_ops::Transpose(transpose_v_params, value_shape, value_data,
                           transpose_v_out_shape, transpose_v_out_data);

  // reshape v for MQA, or add_out (softmax_out)
  if (mqa) {
    TF_LITE_ENSURE_EQ(context, transpose_v_out_tensor->bytes,
                      reshape_v_or_add_out_tensor->bytes);
    memcpy(reshape_v_or_add_out_tensor->data.data,
           transpose_v_out_tensor->data.data, transpose_v_out_tensor->bytes);
  } else {
    // permute softmax_out {0, 1, 3, 2}
    tflite::TransposeParams transpose_softmax_out_params;
    transpose_softmax_out_params.perm_count = 4;
    transpose_softmax_out_params.perm[0] = 0;
    transpose_softmax_out_params.perm[1] = 1;
    transpose_softmax_out_params.perm[2] = 3;
    transpose_softmax_out_params.perm[3] = 2;
    reference_ops::Transpose(transpose_softmax_out_params, add_out_shape,
                             add_out_data, reshape_v_or_add_out_shape,
                             reshape_v_or_add_out_data);
  }

  // mqa FC (softmax_out, squeezed_v)
  // mha BMM(softmax_out, v) transpose_b = true
  if (mqa) {
    tflite::FullyConnectedParams fc_params;
    fc_params.float_activation_min = output_min;
    fc_params.float_activation_max = output_max;
    reference_ops::FullyConnected(fc_params, add_out_shape, add_out_data,
                                  reshape_v_or_add_out_shape,
                                  reshape_v_or_add_out_data, RuntimeShape(),
                                  nullptr, matmul2_out_shape, matmul2_out_data);
  } else {
    // pass rhs first (this is why we transpose add_out above)
    reference_ops::BatchMatMul(
        transpose_v_out_shape, transpose_v_out_data, reshape_v_or_add_out_shape,
        reshape_v_or_add_out_data, matmul2_out_shape, matmul2_out_data);
  }

  // permute out {0, 2, 1, 3}
  tflite::TransposeParams transpose_out_params;
  transpose_out_params.perm_count = 4;
  transpose_out_params.perm[0] = 0;
  transpose_out_params.perm[1] = 2;
  transpose_out_params.perm[2] = 1;
  transpose_out_params.perm[3] = 3;
  reference_ops::Transpose(transpose_out_params, matmul2_out_shape,
                           matmul2_out_data, output_shape, output_data);

  return kTfLiteOk;
}

}  // namespace llm

TfLiteRegistration* Register_SDPA() {
  static TfLiteRegistration r = {llm::SDPAInit, llm::SDPAFree, llm::SDPAPrepare,
                                 llm::SDPAEval};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
