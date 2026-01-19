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

#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <mutex>
#include "mindspore/include/custom_op_api.h"
#include "ops/framework/aclnn/graphmode/aclnn_kernel_mod.h"
#include "ops/framework/utils.h"
#include "ops/c_api/utils/common_utils.h"

namespace ms_custom_ops {

const std::map<std::string, std::string> AscendSocVersionMap = {
  {"Ascend910A", "ascend910"},        {"Ascend910B", "ascend910"},        {"Ascend910PremiumA", "ascend910"},
  {"Ascend910ProA", "ascend910"},     {"Ascend910ProB", "ascend910"},     {"Ascend910B1", "ascend910b"},
  {"Ascend910B2", "ascend910b"},      {"Ascend910B2C", "ascend910b"},     {"Ascend910B3", "ascend910b"},
  {"Ascend910B4", "ascend910b"},      {"Ascend910B4-1", "ascend910b"},    {"Ascend910_9391", "ascend910_93"},
  {"Ascend910_9392", "ascend910_93"}, {"Ascend910_9381", "ascend910_93"}, {"Ascend910_9382", "ascend910_93"},
  {"Ascend910_9372", "ascend910_93"}, {"Ascend910_9361", "ascend910_93"}, {"Ascend910_9362", "ascend910_93"},
  {"Ascend310P", "ascend310p"},       {"Ascend310P3", "ascend310p"},      {"Ascend310B4", "ascend310b"},
  {"Ascend310B1", "ascend310b"},      {"Ascend310", "ascend310"}};

static std::once_flag once_flag;

static std::string GetSocVersion() {
  static std::string soc_version;
  std::call_once(once_flag, [&]() {
    const char *soc_name_ptr = aclrtGetSocName();
    if (soc_name_ptr != nullptr) {
      auto iter = AscendSocVersionMap.find(soc_name_ptr);
      soc_version = (iter != AscendSocVersionMap.end()) ? iter->second : "";
    } else {
      soc_version.clear();
    }
  });
  return soc_version;
}

static bool CheckSocVersion(const std::string &expected_soc_version) {
  const auto &version = GetSocVersion();
  return !version.empty() && (version == expected_soc_version);
}

bool IsSoc910b() { return CheckSocVersion(kAscendVersion910b); }

bool IsSoc910_93() { return CheckSocVersion(kAscendVersion910_93); }

bool IsSoc310p() { return CheckSocVersion(kAscendVersion310p); }
bool IsSoc910a() { return CheckSocVersion(kAscendVersion910); }

bool IsSoc910BC() { return IsSoc910b() || IsSoc910_93(); }

/**
 * @brief Calculate output shape for batch matrix multiplication
 * @param x1_shape Shape of first input tensor
 * @param x2_shape Shape of second input tensor
 * @param transpose_a Whether first tensor is transposed
 * @param transpose_b Whether second tensor is transposed
 * @param offset Number of dimensions to consider for matrix multiplication (typically 2)
 * @param op_name Operator name for error messages
 * @return ShapeVector Output shape vector
 */
ShapeVector BatchMatMulMakeShape(const ShapeVector x1_shape, const ShapeVector x2_shape,
                                 bool transpose_a, bool transpose_b, size_t offset,
                                 const std::string &op_name) {
  constexpr size_t kMatSize = 2;  // Last 2 dimensions for matrix multiplication
  if (x1_shape.size() < kMatSize || x2_shape.size() < kMatSize) {
    MS_LOG(EXCEPTION) << "For '" << op_name << "', the dimension of 'x1' and 'x2' should be at least 2, "
                      << "but got " << x1_shape.size() << " and " << x2_shape.size();
  }

  ShapeVector out_shape;
  ShapeVector long_shape = x1_shape.size() > x2_shape.size() ? x1_shape : x2_shape;
  ShapeVector short_shape = x1_shape.size() > x2_shape.size() ? x2_shape : x1_shape;
  size_t size_diff = long_shape.size() - short_shape.size();

  // Handle batch dimensions (all dimensions except the last 2)
  for (size_t i = 0; i < long_shape.size() - offset; i++) {
    if (long_shape[i] < 0) {
      out_shape.push_back(abstract::Shape::kShapeDimAny);
    } else if (i >= size_diff) {
      // Broadcast: if one dimension is 1, use the other; otherwise they must be equal
      int64_t long_dim = long_shape[i];
      int64_t short_dim = short_shape[i - size_diff];
      if (long_dim == 1) {
        out_shape.push_back(short_dim);
      } else if (short_dim == 1) {
        out_shape.push_back(long_dim);
      } else if (long_dim == short_dim) {
        out_shape.push_back(long_dim);
      } else {
        MS_LOG(EXCEPTION) << "For '" << op_name << "', batch dimensions must be equal or one of them must be 1, "
                          << "but got " << long_dim << " and " << short_dim;
      }
    } else {
      out_shape.push_back(long_shape[i]);
    }
  }

  // Handle matrix dimensions (last 2 dimensions)
  size_t x1_offset = x1_shape.size() - offset;
  size_t x2_offset = x2_shape.size() - offset;

  // Output shape: [..., M, N] where M comes from x1 and N comes from x2
  int64_t x1_m = transpose_a ? x1_shape[x1_offset + 1] : x1_shape[x1_offset];
  int64_t x1_k = transpose_a ? x1_shape[x1_offset] : x1_shape[x1_offset + 1];
  int64_t x2_k = transpose_b ? x2_shape[x2_offset + 1] : x2_shape[x2_offset];
  int64_t x2_n = transpose_b ? x2_shape[x2_offset] : x2_shape[x2_offset + 1];

  // Check K dimensions match
  if (x1_k != abstract::Shape::kShapeDimAny && x2_k != abstract::Shape::kShapeDimAny && x1_k != x2_k) {
    MS_LOG(EXCEPTION) << "For '" << op_name << "', the K dimension of 'x1' and 'x2' must be equal, "
                      << "but got " << x1_k << " and " << x2_k;
  }

  out_shape.push_back(x1_m);
  out_shape.push_back(x2_n);

  return out_shape;
}

/**
 * @brief Check if data type is supported for batch_matmul operation
 * @param dtype Data type to check
 * @param op_name Operator name for error messages
 * 
 * According to aclnn documentation:
 * - Atlas training series products, Atlas inference series products: support FLOAT16, FLOAT32
 * - Atlas A2 training series products/Atlas A2 inference series products,
 *   Atlas A3 training series products/Atlas A3 inference series products: support BFLOAT16, FLOAT16, FLOAT32
 */
void CheckBatchMatMulDataType(TypeId dtype, const std::string &op_name) {
  // Check if it's A2/A3 series chip (910b and 910_93 are A2 series)
  // Note: A3 series detection may need additional chip detection functions
  bool is_a2_a3_series = IsSoc910b() || IsSoc910_93();

  if (is_a2_a3_series) {
    // A2/A3 series support bfloat16, float16, and float32
    if (dtype != TypeId::kNumberTypeFloat16 && dtype != TypeId::kNumberTypeBFloat16 &&
        dtype != TypeId::kNumberTypeFloat32) {
      MS_LOG(EXCEPTION) << "For '" << op_name << "' on Atlas A2/A3 series, only bfloat16, float16, and float32 are "
                        << "supported, but got " << TypeIdToString(dtype);
    }
  } else {
    // Atlas series (non-A2/A3) support float16 and float32
    if (dtype != TypeId::kNumberTypeFloat16 && dtype != TypeId::kNumberTypeFloat32) {
      MS_LOG(EXCEPTION) << "For '" << op_name << "' on Atlas series, only float16 and float32 are supported, "
                        << "but got " << TypeIdToString(dtype) << ". "
                        << "Please use MindSpore built-in BatchMatMul operator for other data types.";
    }
  }
}
}  // namespace ms_custom_ops
