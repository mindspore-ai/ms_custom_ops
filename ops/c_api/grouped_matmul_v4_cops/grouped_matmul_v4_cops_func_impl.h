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

#ifndef MS_CUSTOM_OPS_OPS_C_API_GROUPED_MATMUL_V4_COPS_FUNC_IMPL_H_
#define MS_CUSTOM_OPS_OPS_C_API_GROUPED_MATMUL_V4_COPS_FUNC_IMPL_H_

#include <set>
#include "mindspore/include/custom_op_api.h"
#include "ops/framework/aclnn/graphmode/aclnn_kernel_mod.h"
#include "ops/framework/utils.h"

namespace ms_custom_ops {
constexpr auto kGroupInfo = "group_info";
constexpr size_t kXIndex = 0;
constexpr size_t kWeightIndex = 1;
constexpr size_t kScaleIndex = 3;
constexpr size_t kPerTokenScaleIndex = 7;
constexpr size_t kGroupListIndex = 8;
constexpr size_t kSplitItemIndex = 12;
constexpr size_t kGroupTypeIndex = 13;
constexpr size_t kGroupListTypeIndex = 14;
constexpr size_t kActTypeIndex = 15;
constexpr size_t kWeightFormatIndex = 16;
constexpr size_t kOutDtypeIndex = 17;
constexpr int64_t kGroupTypeOffset = -5;

class GroupedMatmulV4CopsFuncImpl : public OpFuncImpl {
 public:
  GroupedMatmulV4CopsFuncImpl() {}
  ~GroupedMatmulV4CopsFuncImpl() = default;

  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override;
  TypeIdList InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override;
  bool GeneralInferRegistered() const override { return true; };

 protected:
  void FetchGroupInfo(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const;
  int64_t FetchGroupListIndex(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const;
  int64_t FetchGroupListSize(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const;
  ShapeArray InferShapeForSingleOutput(const PrimitivePtr &primitive, const ShapeArray &x_shapes,
                                       const ShapeArray &w_shapes, int64_t group_list_size, int64_t group_type,
                                       bool is_int4) const;
  ShapeArray InferShapeForMultiOutput(const PrimitivePtr &primitive, const ShapeArray &x_shapes,
                                      const ShapeArray &w_shapes) const;
  std::pair<ShapeArray, ShapeArray> FetchInputAndWeightShapes(const PrimitivePtr &primitive,
                                                              const InferInfoPtrList &input_infos) const;
};
}  // namespace ms_custom_ops

#endif  // MS_CUSTOM_OPS_OPS_C_API_GROUPED_MATMUL_V4_COPS_FUNC_IMPL_H_
