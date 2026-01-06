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

#include "ops/c_api/grouped_matmul_v4_cops/grouped_matmul_v4_cops_func_impl.h"
#include <vector>
#include <algorithm>
#include <iterator>
#include <set>
#include <unordered_map>
#include "ops/framework/utils.h"

namespace ms_custom_ops {
constexpr size_t kListTensorInputNums = 12;
constexpr int64_t kGroupTypeMultiOutput = -1;
constexpr int64_t kGroupTypeSingleOutput = 0;

void GroupedMatmulV4CopsFuncImpl::FetchGroupInfo(const PrimitivePtr &primitive,
                                                 const InferInfoPtrList &input_infos) const {
  // for tensortuple(input arg) in backend split. (AscendConvertTupleInputToDynamicInput pass)
  std::vector<int64_t> dyn_input_sizes;
  for (size_t i = 0; i < kListTensorInputNums; i++) {
    const auto &tensors = input_infos[i];
    if (i == LongToSize(kGroupListIndex)) {
      dyn_input_sizes.push_back(1);
      continue;
    }
    if (tensors->IsNone()) {
      dyn_input_sizes.push_back(0);
      continue;
    }
    if (MS_UNLIKELY(tensors->IsDynamicSequence())) {
      MS_EXCEPTION(RuntimeError)
        << "For '" << primitive->name()
        << "', all inputs which is list[tensor] should not be dynamic sequence, which is not supported.";
    }
    const auto &elements = tensors->GetSequenceElements();
    dyn_input_sizes.push_back(SizeToLong(elements.size()));
  }
  primitive->set_attr(kGroupInfo, MakeValue(dyn_input_sizes));  // len of tuple input
}

int64_t GroupedMatmulV4CopsFuncImpl::FetchGroupListIndex(const PrimitivePtr &primitive,
                                                         const InferInfoPtrList &input_infos) const {
  if (MS_LIKELY(input_infos[kXIndex]->IsSequence())) {
    return kGroupListIndex;
  }
  // Runtime phase: the element in input_args is KernelTensor. (tuple is expanded)
  const auto &group_info = GetValue<std::vector<int64_t>>(primitive->GetAttr(kGroupInfo));
  std::vector<int64_t> start_idxes{0};
  int64_t cur_end_idx = 0;
  for (size_t i = 0; i < group_info.size(); ++i) {
    cur_end_idx += (group_info[i] == 0 ? 1 : group_info[i]);
    start_idxes.push_back(cur_end_idx);
  }
  return start_idxes.at(kGroupListIndex);
}

ShapeArray GroupedMatmulV4CopsFuncImpl::InferShapeForSingleOutput(const PrimitivePtr &primitive,
                                                                  const ShapeArray &x_shapes,
                                                                  const ShapeArray &w_shapes, int64_t group_list_size,
                                                                  int64_t group_type, bool is_int4) const {
  if (MS_UNLIKELY(x_shapes.size() != kDim1 || w_shapes.size() != kDim1)) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', when split_item is 3. the size of x and weight should both be 1, but got x's size "
                             << x_shapes.size() << ", and weight's size " << w_shapes.size();
  }

  const auto &x_shape = x_shapes[0];
  const auto &w_shape = w_shapes[0];
  auto is_x_dyn_rank = IsDynamicRank(x_shape);
  auto is_w_dyn_rank = IsDynamicRank(w_shape);
  auto m = is_x_dyn_rank ? abstract::Shape::kShapeDimAny : x_shape[x_shape.size() - 2];
  auto n = abstract::Shape::kShapeDimAny;
  if (!is_w_dyn_rank) {
    n = w_shape.back();
    if (is_int4) {
      n = n << 1;
    }
  }

  std::vector<int64_t> res_shape;
  if (group_type == kGroupTypeSingleOutput) {
    // x.shape [m, k], w.shape [e, k, n], y.shape [m, n]
    res_shape = std::vector<int64_t>{m, n};
  } else {
    // x.shape [m, k], w.shape [k, n], y.shape [b, m, n]
    res_shape = std::vector<int64_t>{group_list_size, m, n};
  }
  return {std::move(res_shape)};
}

ShapeArray GroupedMatmulV4CopsFuncImpl::InferShapeForMultiOutput(const PrimitivePtr &primitive,
                                                                 const ShapeArray &x_shapes,
                                                                 const ShapeArray &w_shapes) const {
  if (MS_UNLIKELY(x_shapes.size() != w_shapes.size())) {
    MS_EXCEPTION(ValueError)
      << "For '" << primitive->name()
      << "', when group_type is -1 and split_item is 0, x's size should be equal to weight, but got ."
      << x_shapes.size() << " and " << w_shapes.size();
  }

  ShapeArray output_shapes;
  for (size_t i = 0; i < x_shapes.size(); i++) {
    const auto &x_shape = x_shapes[i];
    const auto &w_shape = w_shapes[i];
    if (MS_UNLIKELY(IsDynamicRank(x_shape))) {
      (void)output_shapes.emplace_back(ShapeVector{abstract::TensorShape::kShapeRankAny});
    } else {
      auto res_shape = x_shape;
      res_shape.back() = IsDynamicRank(w_shape) ? abstract::Shape::kShapeDimAny : w_shape.back();
      (void)output_shapes.emplace_back(std::move(res_shape));
    }
  }
  return output_shapes;
}

int64_t GroupedMatmulV4CopsFuncImpl::FetchGroupListSize(const PrimitivePtr &primitive,
                                                        const InferInfoPtrList &input_infos) const {
  const auto group_list_idx = FetchGroupListIndex(primitive, input_infos);
  const auto &group_list_shape = input_infos.at(group_list_idx)->GetShape();
  MS_CHECK_VALUE(group_list_shape.size() == kDim1,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("group_list's rank", group_list_shape.size(), kEqual,
                                                             kDim1, primitive));
  return input_infos[group_list_idx]->IsDynamic() ? abstract::Shape::kShapeDimAny : group_list_shape[kIndex0];
}

std::pair<ShapeArray, ShapeArray> GroupedMatmulV4CopsFuncImpl::FetchInputAndWeightShapes(
  const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  ShapeArray x_shapes;
  ShapeArray w_shapes;
  if (MS_LIKELY(input_infos[kXIndex]->IsSequence())) {
    FetchGroupInfo(primitive, input_infos);
    auto FetchTupleTensorShapeFunc = [](const InferInfoPtr &tensors) {
      const auto &elements = tensors->GetSequenceElements();
      ShapeArray shapes;
      std::transform(elements.begin(), elements.end(), std::back_inserter(shapes),
                     [](const InferInfoPtr &info) { return info->GetShape(); });
      return shapes;
    };
    // get tuple_x_shape in compile phase
    x_shapes = FetchTupleTensorShapeFunc(input_infos[kXIndex]);
    // get tuple_w_shape in compile phase
    w_shapes = FetchTupleTensorShapeFunc(input_infos[kWeightIndex]);
  } else {
    // Runtime phase: the element in input_args is KernelTensor. (tuple is expanded)
    auto tuple_len = GetValue<std::vector<int64_t>>(primitive->GetAttr(kGroupInfo));
    size_t x_idx_end = LongToSize(tuple_len[kXIndex]);
    size_t w_idx_end = LongToSize(tuple_len[kXIndex] + tuple_len[kWeightIndex]);
    std::transform(input_infos.begin(), input_infos.begin() + x_idx_end, std::back_inserter(x_shapes),
                   [](const InferInfoPtr &info) { return info->GetShape(); });
    std::transform(input_infos.begin() + x_idx_end, input_infos.begin() + w_idx_end, std::back_inserter(w_shapes),
                   [](const InferInfoPtr &info) { return info->GetShape(); });
  }
  return std::make_pair(std::move(x_shapes), std::move(w_shapes));
}

ShapeArray GroupedMatmulV4CopsFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                   const InferInfoPtrList &input_infos) const {
  auto [x_shapes, w_shapes] = FetchInputAndWeightShapes(primitive, input_infos);
  const auto input_num = SizeToLong(input_infos.size());
  auto group_type_index = input_num + kGroupTypeOffset;
  auto group_type = input_infos[group_type_index]->GetScalarValueWithCheck<int64_t>();
  if (group_type == kGroupTypeMultiOutput) {
    return InferShapeForMultiOutput(primitive, x_shapes, w_shapes);
  }

  auto group_list_size = FetchGroupListSize(primitive, input_infos);
  bool is_int4 = false;
  if (MS_LIKELY(input_infos[kWeightIndex]->IsSequence())) {
    const auto &w_tensors = input_infos[kWeightIndex]->GetSequenceElements();
    MS_CHECK_VALUE(w_tensors.size() > 0, "For 'grouped_matmul_v4_cops', 'weight' must be provided.");
    is_int4 = w_tensors[0]->GetType() == kNumberTypeInt4;
  } else {
    is_int4 = input_infos[kWeightIndex]->GetType() == kNumberTypeInt4;
  }

  return InferShapeForSingleOutput(primitive, x_shapes, w_shapes, group_list_size, group_type, is_int4);
}

TypeIdList GroupedMatmulV4CopsFuncImpl::InferType(const PrimitivePtr &primitive,
                                                  const InferInfoPtrList &input_infos) const {
  const auto &x_tensors = input_infos[kXIndex]->GetSequenceElements();
  const auto &w_tensors = input_infos[kWeightIndex]->GetSequenceElements();
  const auto &scale_infos = input_infos[kScaleIndex];
  TypeId x_type = x_tensors[0]->GetType();
  TypeId w_type = w_tensors[0]->GetType();
  TypeIdList output_types;
  if (x_type == kNumberTypeInt8 && w_type == kNumberTypeInt4) {
    TypeId output_type = kNumberTypeBFloat16;
    if (!input_infos[kOutDtypeIndex]->IsNone()) {
      auto dtype_ptr = input_infos[kOutDtypeIndex]->GetScalarValueWithCheck<int64_t>();
      output_type = static_cast<TypeId>(dtype_ptr);
      static std::set<TypeId> valid_dtype_set = {kNumberTypeFloat16, kNumberTypeBFloat16};
      MS_CHECK_VALUE(
        valid_dtype_set.find(output_type) != valid_dtype_set.end(),
        "For 'grouped_matmul_v4_cops' with A8W4, the output type must be in [Float16, BFloat16], but got " +
          TypeIdToString(output_type));
    }
    std::transform(x_tensors.begin(), x_tensors.end(), std::back_inserter(output_types),
                   [output_type](const InferInfoPtr &info) { return output_type; });
  } else if (scale_infos->IsNone()) {
    auto out_type = x_type == kNumberTypeInt8 ? kNumberTypeInt32 : x_type;
    std::transform(x_tensors.begin(), x_tensors.end(), std::back_inserter(output_types),
                   [=](const InferInfoPtr &info) { return out_type; });
  } else {
    const auto &scale_tensors = scale_infos->GetSequenceElements();
    TypeId scale_type = scale_tensors[0]->GetType();
    if (scale_type == kNumberTypeUInt64) {
      std::transform(x_tensors.begin(), x_tensors.end(), std::back_inserter(output_types),
                     [](const InferInfoPtr &info) { return kNumberTypeInt8; });
    } else if (scale_type == kNumberTypeBFloat16) {
      std::transform(x_tensors.begin(), x_tensors.end(), std::back_inserter(output_types),
                     [](const InferInfoPtr &info) { return kNumberTypeBFloat16; });
    } else if (scale_type == kNumberTypeFloat32) {
      std::transform(x_tensors.begin(), x_tensors.end(), std::back_inserter(output_types),
                     [](const InferInfoPtr &info) { return kNumberTypeFloat16; });
    } else {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                               << "', the scale only support Uint16, BFloat16 and Float32, but got " << scale_type;
    }
  }
  return output_types;
}
}  // namespace ms_custom_ops
