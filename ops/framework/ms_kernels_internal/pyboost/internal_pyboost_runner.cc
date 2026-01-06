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

#include "ops/framework/ms_kernels_internal/pyboost/internal_pyboost_runner.h"

namespace {
constexpr size_t kMemAlignSize{512};
constexpr size_t kAlignBytes{32};
size_t GetAlignedSize(size_t size) {
  return (size + kMemAlignSize + kAlignBytes -1) / kMemAlignSize * kMemAlignSize;
}
}

namespace ms_custom_ops {
void InternalPyboostRunner::GetOrCreateKernel(const TensorList &inputs, const TensorList &outputs) {
  // Disable cross-runner InternalOp caching: always create a fresh InternalOp instance
  MS_LOG(DEBUG) << "Internal Op [" << this->op_name() << "] create new instance";
  TransDataType(inputs, outputs);
  UpdateArgImmutableInfo(&inputs_ii_, inputs, true);
  UpdateArgImmutableInfo(&outputs_ii_, outputs);
  internal_op_ = CreateKernel(inputs_ii_, outputs_ii_);
  MS_EXCEPTION_IF_NULL(internal_op_);
  auto status = internal_op_->Init();
  if (status != mindspore::internal_v2::kInternalOk) {
    internal_op_ = nullptr;
    MS_LOG(EXCEPTION) << "Init internal kernel failed, kernel_name: " << this->op_name();
    return;
  }

  internal_inputs_shape_.clear();
  internal_outputs_shape_.clear();
  internal_inputs_shape_.resize(inputs.size());
  internal_outputs_shape_.resize(outputs.size());
  TransInternalShapes(&internal_inputs_shape_, inputs, true);
  TransInternalShapes(&internal_outputs_shape_, outputs, false);

  if (!UpdateParam()) {
    MS_LOG(EXCEPTION) << "UpdateParam failed, kernel_name: " << this->op_name();
  }
  auto internal_ret = internal_op_->UpdateShape(internal_inputs_shape_, internal_outputs_shape_);
  if (internal_ret != mindspore::internal_v2::kInternalOk) {
    MS_LOG(EXCEPTION) << "InternalKernel UpdateShape failed, kernel_name: " << this->op_name();
  }

  tiling_cache_item_ = GetOrGenerateTiling();
}

size_t InternalPyboostRunner::CalcWorkspace() {
  MS_EXCEPTION_IF_NULL(internal_op_);
  // Workspace size is derived from tiling cache (workspace_size_list_),
  // decoupled from the current InternalOp instance lifetime.
  return std::accumulate(workspace_size_list_.begin(), workspace_size_list_.end(), 0,
                         [](size_t acc, size_t size){
                           return acc + GetAlignedSize(size);
                         });
}

void InternalPyboostRunner::TransDataType(const TensorList &ms_inputs, const TensorList &ms_outputs) {
  internal_inputs_dtype_.resize(ms_inputs.size());
  internal_outputs_dtype_.resize(ms_outputs.size());

  for (size_t i = 0; i < ms_inputs.size(); ++i) {
    if (!ms_inputs[i].is_defined()) {
      internal_inputs_dtype_[i] = mindspore::internal_v2::DataType::kTypeNone;
      continue;
    }

    internal_inputs_dtype_[i] = TransInternalDataType(ms_inputs[i].data_type());
  }

  for (size_t i = 0; i < ms_outputs.size(); ++i) {
    if (!ms_outputs[i].is_defined()) {
      internal_outputs_dtype_[i] = mindspore::internal_v2::DataType::kTypeNone;
      continue;
    }
    internal_outputs_dtype_[i] = TransInternalDataType(ms_outputs[i].data_type());
  }
}

TilingCacheItemPtr InternalPyboostRunner::GetOrGenerateTiling() {
  std::lock_guard<SimpleSpinLock> lock(lock_);
  auto key = GetOrGenerateOpTilingKey(tiling_key_);
  auto tiling_info_ptr = InternalTilingCache::GetInstance().Bind(key);
  if (tiling_info_ptr == nullptr) {
    MS_LOG(INFO) << "start create tiling info for " << this->op_name();
    auto tiling_size = internal_op_->GetTilingSize();
    auto host_addr = TilingMemMgr::GetInstance().pool_host_.Malloc(tiling_size);
    mindspore::internal_v2::HostRunInfoPtr host_run_info_ptr = nullptr;
    auto status = internal_op_->Tiling(host_addr, &host_run_info_ptr);
    if (status != mindspore::internal_v2::kInternalOk || host_run_info_ptr == nullptr) {
      MS_LOG(EXCEPTION) << "Tiling error for " << this->op_name() << ", status: " << status
                        << ", host_run_info_ptr: " << host_run_info_ptr;
    }
    auto device_addr = TilingMemMgr::GetInstance().pool_device_.Malloc(tiling_size);
    TilingMemMgr::GetInstance().CopyAsync(host_addr, device_addr, tiling_size);
    auto tiling_info = std::make_shared<mindspore::internal_v2::TilingInfo>(device_addr, nullptr);
    tiling_info->host_run_info_ = host_run_info_ptr;
    // Cache workspace size into tiling info and runner-local workspace_size_list_
    workspace_size_list_ = internal_op_->GetWorkspaceSize();
    tiling_info->host_run_info_->SetWorkSpaceSize(workspace_size_list_);
    tiling_info_ptr = std::make_shared<TilingCacheItem>(tiling_info, host_addr, tiling_size);
    if (TilingMemMgr::GetInstance().pool_device_.IsOneOffMem(device_addr)) {
      // tiling mem pool is full, comb out some items which are not recently
      // used with high probability
      auto erased_items = InternalTilingCache::GetInstance().CombOutSuspectedUselessItems();
      if (!erased_items.empty()) {
        for (auto &item : erased_items) {
          TilingMemMgr::GetInstance().pool_device_.Free(item->tiling_info_->tiling_addr_, item->size_);
          TilingMemMgr::GetInstance().pool_host_.Free(item->host_addr_, item->size_);
        }
        TilingMemMgr::GetInstance().pool_device_.Rearrange();
        TilingMemMgr::GetInstance().pool_host_.Rearrange();
      }
      MS_LOG(INFO) << "The tiling memory pool is full, comb out not used items: " << erased_items.size();
    }
    (void)InternalTilingCache::GetInstance().Insert(key, tiling_info_ptr);
    MS_LOG(INFO) << "end create tiling info for " << this->op_name();
  } else {
    // Hit tiling cache: restore workspace size from cached HostRunInfo
    if (tiling_info_ptr->tiling_info_ != nullptr &&
        tiling_info_ptr->tiling_info_->host_run_info_ != nullptr) {
      workspace_size_list_ = tiling_info_ptr->tiling_info_->host_run_info_->GetWorkSpaceSize();
    } else {
      workspace_size_list_.clear();
    }
  }
  return tiling_info_ptr;
}

void InternalPyboostRunner::TransInternalShapes(mindspore::internal_v2::ShapeInfoList *shapelist,
                                                const TensorList &tensorlist, bool is_input) {
  for (size_t i = 0; i < tensorlist.size(); i++) {
    if (!tensorlist[i].is_defined()) {
      shapelist->at(i) = mindspore::internal_v2::ShapeInfo{};
      continue;
    }

    if (!tensorlist[i].is_contiguous()) {
      if (is_input) {
        MS_LOG(EXCEPTION) << "For internal op [" << this->op_name() << "], the input tensorlist[" << i
                          << "] is not contiguous: "
                          << ", please convert it to contiguous tensor using "
                             "tensor.contiguous().";
      } else {
        MS_LOG(EXCEPTION) << "For internal op [" << this->op_name() << "], the output tensorlist[" << i
                          << "] is not contiguous: "
                          << ", please convert it to contiguous tensor using "
                             "tensor.contiguous().";
      }
    }

    auto shape = tensorlist[i].data_type() != kMetaTypeNone ? TransInternalShape(tensorlist[i].shape())
                                                            : mindspore::internal_v2::ShapeInfo{0};
    shapelist->at(i) = std::move(shape);
  }
}

void InternalPyboostRunner::UpdateArgImmutableInfo(internal_v2::ArgImmutableInfo *arginfo, const ms::Tensor &tensor,
                                                   internal_v2::DataType dtype) {
  arginfo->SetDtype(dtype);
  if (!tensor.is_defined()) {
    arginfo->SetFormat(internal_v2::TensorFormat::kFormatND);
    return;
  }
  arginfo->SetFormat(TransInternalFormat(GetFormatFromStrToEnum(tensor.format())));
}

void InternalPyboostRunner::UpdateArgImmutableInfo(std::vector<internal_v2::ArgImmutableInfo> *arginfos,
                                                   const TensorList &tensorlist, bool is_input) {
  arginfos->resize(tensorlist.size());
  for (size_t i = 0; i < tensorlist.size(); ++i) {
    if (is_input) {
      UpdateArgImmutableInfo(&(arginfos->at(i)), tensorlist[i], internal_inputs_dtype_[i]);
    } else {
      UpdateArgImmutableInfo(&(arginfos->at(i)), tensorlist[i], internal_outputs_dtype_[i]);
    }
  }
}

void InternalPyboostRunner::GetWorkspace(const internal_v2::InternalOpPtr &internal_op,
                                         internal_v2::WsAddrList *internal_wss_addr) {
  auto workspace_ptr = this->workspace_ptr();
  if (workspace_ptr == nullptr) {
    return;
  }
  MS_EXCEPTION_IF_NULL(internal_op);
  internal_wss_addr->resize(workspace_size_list_.size());

  size_t offset = 0;
  for (size_t i = 0; i < workspace_size_list_.size(); i++) {
    auto work_ptr = static_cast<void *>(static_cast<int8_t *>(workspace_ptr) + offset);
    internal_wss_addr->at(i) = work_ptr;
    offset += GetAlignedSize(workspace_size_list_[i]);
  }
}

void InternalPyboostRunner::LaunchKernel() {
  MS_EXCEPTION_IF_NULL(tiling_cache_item_);
  MS_EXCEPTION_IF_NULL(internal_op_);
  internal_v2::InputsAddrList inputs_addr;
  internal_v2::OutputsAddrList outputs_addr;
  InternalPyboostRunner::UpdateAddr(&inputs_addr, this->inputs());
  InternalPyboostRunner::UpdateAddr(&outputs_addr, this->outputs());
  internal_v2::WsAddrList _internal_wss_addr;
  InternalPyboostRunner::GetWorkspace(internal_op_, &_internal_wss_addr);

  auto op_name = this->op_name();
  MS_LOG(DEBUG) << "Launch InternalKernel " << op_name << " start";
  internal_op_->SetTilingInfo(tiling_cache_item_->tiling_info_);
  auto &internal_wss_addr = const_cast<internal_v2::WsAddrList &>(_internal_wss_addr);
  internal_v2::InternalStatus status =
    internal_op_->Launch(inputs_addr, outputs_addr, internal_wss_addr, this->stream(), op_name);
  InternalTilingCache::GetInstance().Unbind(tiling_cache_item_);
  if (status != internal_v2::InternalStatus::kInternalOk) {
    MS_LOG(EXCEPTION) << "Launch InternalKernel failed, kernel_name: " << op_name;
  }
  MS_LOG(DEBUG) << "Launch InternalKernel " << op_name << " end";
}
}  // namespace ms_custom_ops
