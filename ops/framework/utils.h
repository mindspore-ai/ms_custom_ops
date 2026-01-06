/**
 * Copyright 2025 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __MS_CUSTOM_OPS_CCSRC_UTILS_UTILS_H__
#define __MS_CUSTOM_OPS_CCSRC_UTILS_UTILS_H__

#include <pybind11/pybind11.h>
#include <cstdint>
#include <string>
#include <vector>
#include "mindspore/include/custom_op_api.h"

namespace ms_custom_ops {
constexpr size_t kIndex0{0};
constexpr size_t kIndex1{1};
constexpr size_t kIndex2{2};
constexpr size_t kIndex3{3};
constexpr size_t kIndex4{4};
constexpr size_t kIndex5{5};
constexpr size_t kIndex6{6};
constexpr size_t kIndex7{7};
constexpr size_t kIndex8{8};
constexpr size_t kIndex9{9};
constexpr size_t kDim0{0};
constexpr size_t kDim1{1};
constexpr size_t kDim2{2};
constexpr size_t kDim3{3};
constexpr size_t kDim4{4};
constexpr size_t kDim5{5};
constexpr size_t kDim6{6};
constexpr size_t kDim7{7};
constexpr size_t kDim8{8};

constexpr size_t kNumber0{0};
constexpr size_t kNumber1{1};
constexpr size_t kNumber2{2};
constexpr size_t kNumber3{3};
constexpr size_t kNumber4{4};
constexpr size_t kNumber5{5};
constexpr size_t kNumber6{6};

constexpr auto kFractalNzFormat = "FRACTAL_NZ";
constexpr auto kNdFormat = "ND";
// 用于静态图下没有返回值的算子InferShape占位输出
static const mindspore::ShapeArray kFakeOutTensorShapes{mindspore::ShapeVector{1}};
// 用于静态图下没有返回值的算子InferType占位输出
static const std::vector<mindspore::TypeId> kFakeOutTensorTypes{mindspore::TypeId::kNumberTypeInt8};
static const mindspore::ShapeVector kFakeOutShapes{1};

// Helper function to convert optional tensor to tensor or empty tensor
inline ms::Tensor GetTensorOrEmpty(const std::optional<ms::Tensor> &opt_tensor) {
  return opt_tensor.has_value() ? opt_tensor.value() : ms::Tensor();
}

inline void *GetHostDataPtr(const ms::Tensor &tensor) {
  auto tensor_ptr = tensor.tensor();
  MS_EXCEPTION_IF_NULL(tensor_ptr);
  return tensor_ptr->data_c();
}

template <typename T, mindspore::TypeId DATA_TYPE>
T *GetRawPtr(const ms::Tensor &tensor, const std::string &op_name, const std::string &tensor_name) {
  if (tensor.data_type() != DATA_TYPE) {
    MS_LOG(EXCEPTION) << "For " << op_name << ", the data_type of " << tensor_name << " must be " << DATA_TYPE
                      << ", but got: " << tensor.data_type();
  }

  auto ptr = GetHostDataPtr(tensor);
  if (ptr == nullptr) {
    MS_LOG(EXCEPTION) << "For " << op_name << ", the data ptr of " << tensor_name << " can not be nullptr.";
  }
  return reinterpret_cast<T *>(ptr);
}

template <typename T, mindspore::TypeId DATA_TYPE>
inline std::vector<T> GetVectorFromTensor(const ms::Tensor &tensor, const std::string &op_name,
                                          const std::string &tensor_name) {
  auto vptr = GetRawPtr<T, DATA_TYPE>(tensor, op_name, tensor_name);
  return std::vector<T>(vptr, vptr + tensor.numel());
}

template <typename T>
T GetValueFromTensor(const ms::Tensor &tensor, const std::string &op_name, const std::string &tensor_name) {
  if constexpr (std::is_same_v<T, std::vector<int32_t>>) {
    return GetVectorFromTensor<int32_t, mindspore::kNumberTypeInt32>(tensor, op_name, tensor_name);
  }

  if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
    return GetVectorFromTensor<int64_t, mindspore::kNumberTypeInt64>(tensor, op_name, tensor_name);
  }

  MS_LOG(EXCEPTION) << "Not implemented. op_name: " << op_name << ", tensor_name: " << tensor_name
                    << ", type: " << typeid(T).name();
}

// ============================================================================
// FRACTAL_NZ Format Common Definitions
// ============================================================================
// These constants and enums are shared across multiple operators that work
// with FRACTAL_NZ format (e.g., trans_data, reshape_and_cache, mla, etc.)

// TransData format conversion types
enum class TransDataFormat : int32_t {
  FRACTAL_NZ_TO_ND = 0,
  ND_TO_FRACTAL_NZ = 1,
};

// format types
enum class DataFormat : int32_t {
  ND = 0,
  FRACTAL_NZ = 1,
};

// Alignment constants for FRACTAL_NZ format
constexpr int64_t kNzHeightAlign = 16;
constexpr int64_t kNzWidthAlignDefault = 16;  // For fp16/bf16
constexpr int64_t kNzWidthAlignInt8 = 32;     // For int8/uint8

// Align dimension to the specified boundary
inline int64_t AlignDimension(int64_t dim, int64_t align_boundary) {
  return ((dim + align_boundary - 1) / align_boundary) * align_boundary;
}

// Check that dimension is aligned to the specified boundary
// Throws exception if dimension is not properly aligned
inline void CheckDimensionAlignment(int64_t dim, int64_t align_boundary, const std::string &dim_name) {
  int64_t remainder = dim % align_boundary;
  if (remainder != 0) {
    MS_LOG(EXCEPTION) << "Input " << dim_name << " dimension must be aligned to " << align_boundary << ", but got "
                      << dim << " (remainder: " << remainder << ")";
  }
}

// Validate shape's last two dimensions (H, W) are properly aligned for FRACTAL_NZ format
inline bool CheckShapeHWAlignment(const mindspore::ShapeVector &shape, mindspore::TypeId data_type) {
  if (shape.size() < kNumber2) {
    MS_LOG(EXCEPTION) << "Shape must have at least 2 dimensions, but got " << shape.size();
  }
  int64_t h_dim = shape[shape.size() - kNumber2];
  int64_t w_dim = shape[shape.size() - kNumber1];

  CheckDimensionAlignment(h_dim, kNzHeightAlign, "H");
  int64_t w_align = (data_type == mindspore::kNumberTypeInt8 || data_type == mindspore::kNumberTypeUInt8)
                      ? kNzWidthAlignInt8
                      : kNzWidthAlignDefault;
  CheckDimensionAlignment(w_dim, w_align, "W");
  return true;
}

inline mindspore::TensorStorageInfoPtr GetNZFormatStorageInfo(const mindspore::ShapeVector &nd_shape,
                                                              const mindspore::TypeId &dtype_id) {
  auto nz_shape = mindspore::trans::DeviceShapeTransfer().GetDeviceShapeByFormat(nd_shape, kFractalNzFormat, dtype_id);

  constexpr int64_t kStrideBase = 1;
  constexpr int kStrideOffset = 2;
  auto strides = nd_shape;
  if (!strides.empty()) {
    strides.erase(strides.begin());
  }
  strides.push_back(kStrideBase);
  for (int i = static_cast<int>(strides.size()) - kStrideOffset; i >= 0; i--) {
    strides[i] = strides[i] * strides[i + 1];
  }
  auto storage_info = std::make_shared<mindspore::TensorStorageInfo>(nd_shape, strides, nz_shape, strides, true);
  return storage_info;
}

class GilReleaseWithCheck {
 public:
  GilReleaseWithCheck() {
    if (Py_IsInitialized() != 0 && PyGILState_Check() != 0) {
      release_ = std::make_unique<pybind11::gil_scoped_release>();
    }
  }

  ~GilReleaseWithCheck() {
    release_ = nullptr;
  }
 private:
  std::unique_ptr<pybind11::gil_scoped_release> release_;
};
}  // namespace ms_custom_ops
#endif  // __MS_CUSTOM_OPS_CCSRC_UTILS_UTILS_H__
