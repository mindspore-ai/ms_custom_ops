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
#ifdef __DAV_C220_VEC__
#include "ascendc/basic_api/kernel_operator.h"

static constexpr int32_t BLOCK_SIZE = 32;

template <bool kv_cache_npd, bool kv_npd, typename Dtype>
class ReshapeAndCache {
 public:
  void __aicore__ __inline__ Init(GM_ADDR key, GM_ADDR value, GM_ADDR key_cache, GM_ADDR value_cache,
                                  GM_ADDR slot_mapping, GM_ADDR q_seq, GM_ADDR kv_seq, GM_ADDR block_tbl, GM_ADDR k_out,
                                  GM_ADDR v_out, GM_ADDR workspace, GM_ADDR tiling) {
    in_key_.SetGlobalBuffer(reinterpret_cast<__gm__ Dtype *>(key));
    in_value_.SetGlobalBuffer(reinterpret_cast<__gm__ Dtype *>(value));
    in_out_key_cache_.SetGlobalBuffer(reinterpret_cast<__gm__ Dtype *>(key_cache));
    in_out_value_cache_.SetGlobalBuffer(reinterpret_cast<__gm__ Dtype *>(value_cache));
    in_slot_mapping_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(slot_mapping));
    in_block_tbl_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(block_tbl));
    out_key_.SetGlobalBuffer(reinterpret_cast<__gm__ Dtype *>(k_out));
    out_value_.SetGlobalBuffer(reinterpret_cast<__gm__ Dtype *>(v_out));
    in_tiling.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(tiling));

    num_tokens_ = in_tiling.GetValue(2);
    hidden_size_ = in_tiling.GetValue(3);
    k_head_num_ = in_tiling.GetValue(4);
    v_head_num_ = in_tiling.GetValue(5);
    page_size_ = in_tiling.GetValue(6);
    batch_size_ = in_tiling.GetValue(7);
    uint32_t allc_size = (k_head_num_ > v_head_num_ ? k_head_num_ : v_head_num_) * hidden_size_;
    in_local_ = ub_allocator_.Alloc<Dtype>(allc_size);
    in_kv_seq_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(tiling) + 32);
    in_q_seq_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(tiling) + 32 + batch_size_);
  }

  void __aicore__ __inline__ Process() {
    int32_t core_num = get_block_num();
    int64_t per_core_task_num = num_tokens_ * 2 / core_num;
    int32_t tail_task_num = num_tokens_ * 2 % core_num;

    int32_t block_id = get_block_idx();
    int64_t start_task_id = block_id * per_core_task_num;

    if (block_id < tail_task_num) {
      per_core_task_num++;
      start_task_id += block_id;
    } else {
      start_task_id += tail_task_num;
    }

    for (int64_t i = 0; i < per_core_task_num; i++) {
      if (i + start_task_id < num_tokens_) {
        ReshapeAndCacheDo(in_key_, k_head_num_, hidden_size_ / k_head_num_, (i + start_task_id), in_out_key_cache_,
                          out_key_);
      } else {
        ReshapeAndCacheDo(in_value_, v_head_num_, hidden_size_ / v_head_num_, (i + start_task_id - num_tokens_),
                          in_out_value_cache_, out_value_);
      }
    }
  }

 private:
  void __aicore__ __inline__ CopyIn(AscendC::GlobalTensor<Dtype> &in_kv, uint64_t src_offset,
                                    AscendC::DataCopyParams cpy_param) {
    DataCopy(in_local_, in_kv[src_offset], cpy_param);
  }

  void __aicore__ __inline__ CopyOut(AscendC::GlobalTensor<Dtype> &out, uint64_t dst_offset,
                                     AscendC::DataCopyParams cpy_param) {
    DataCopy(out[dst_offset], in_local_, cpy_param);
  }

  void __aicore__ __inline__ ComputeKvNpdOffset(int token_idx, int head_num, int embed, uint64_t *token_offset) {
    uint32_t offset = 0;
    uint32_t i = 0;
    int32_t limit = in_kv_seq_.GetValue(i);
    int32_t prev_seq_len = limit;
    uint32_t prev_token = 0;
    while (token_idx >= limit) {
      offset += ((prev_seq_len + page_size_ - 1) / page_size_) * page_size_;
      prev_token += prev_seq_len;
      i++;
      prev_seq_len = in_kv_seq_.GetValue(i);
      limit += prev_seq_len;
    }
    uint32_t inner_token_idx = token_idx - prev_token;
    uint32_t batch_offset = offset * head_num * embed;
    uint32_t inner_batch_offset =
      (inner_token_idx / page_size_) * head_num * page_size_ * embed + (inner_token_idx % page_size_) * embed;
    *token_offset = batch_offset + inner_batch_offset;
  }

  void __aicore__ __inline__ ReshapeAndCacheDo(AscendC::GlobalTensor<Dtype> &in_kv, int32_t kv_head_num, int32_t embed,
                                               int32_t token_idx, AscendC::GlobalTensor<Dtype> &out_cache,
                                               AscendC::GlobalTensor<Dtype> &out_kv) {
    uint64_t start = token_idx * hidden_size_;
    int32_t slot_value = in_slot_mapping_.GetValue(token_idx);
    if (slot_value < 0) return;
    uint16_t num_hidden_blocks = hidden_size_ * sizeof(Dtype) / BLOCK_SIZE;
    uint16_t num_embed_blocks = embed * sizeof(Dtype) / BLOCK_SIZE;

    // step I: copy token into local memory
    AscendC::DataCopyParams cpy_in_param(1, num_hidden_blocks, 0, 0);
    CopyIn(in_kv, start, cpy_in_param);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
    // step II: copy token into kv_cache
    // cache is stored [B_NUM, B_SIZE, N, D]
    uint64_t cache_start = slot_value * hidden_size_;
    uint16_t dst_stride = 0;
    uint16_t n_burst = 1;
    uint16_t n_blks = num_hidden_blocks;
    if constexpr (kv_cache_npd) {
      // cache is stored [B_NUM, N, B_SIZE, D]
      cache_start = (slot_value / page_size_) * kv_head_num * page_size_ * embed + (slot_value % page_size_) * embed;
      dst_stride = (page_size_ - 1) * embed * sizeof(Dtype) / BLOCK_SIZE;
      n_burst = kv_head_num;
      n_blks = num_embed_blocks;
    }
    AscendC::DataCopyParams cpy_out_cache_param(n_burst, n_blks, 0, dst_stride);
    CopyOut(out_cache, cache_start, cpy_out_cache_param);

    // step III: copy token into kv_out
    // update key\value - [TND]
    uint64_t token_start = start;
    dst_stride = 0;
    n_burst = 1;
    n_blks = num_hidden_blocks;
    if constexpr (kv_npd) {
      // update key\value - [T/B_SIZE, N, B_SIZE, D]
      dst_stride = (page_size_ - 1) * embed * sizeof(Dtype) / BLOCK_SIZE;
      n_burst = kv_head_num;
      n_blks = num_embed_blocks;
      ComputeKvNpdOffset(token_idx, kv_head_num, embed, &token_start);
    }
    AscendC::DataCopyParams cpy_out_kv_param(n_burst, n_blks, 0, dst_stride);
    CopyOut(out_kv, token_start, cpy_out_kv_param);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    return;
  }
  AscendC::LocalMemAllocator<AscendC::Hardware::UB> ub_allocator_;
  AscendC::GlobalTensor<Dtype> in_key_;
  AscendC::GlobalTensor<Dtype> in_value_;
  AscendC::GlobalTensor<Dtype> in_out_key_cache_;
  AscendC::GlobalTensor<Dtype> in_out_value_cache_;
  AscendC::GlobalTensor<int32_t> in_slot_mapping_;
  AscendC::GlobalTensor<int32_t> in_q_seq_;
  AscendC::GlobalTensor<int32_t> in_kv_seq_;
  AscendC::GlobalTensor<int32_t> in_block_tbl_;
  AscendC::GlobalTensor<int32_t> in_tiling;
  AscendC::GlobalTensor<Dtype> out_key_;
  AscendC::GlobalTensor<Dtype> out_value_;
  AscendC::LocalTensor<Dtype> in_local_;

  int32_t num_tokens_ = 0;
  int32_t hidden_size_ = 0;
  int32_t k_head_num_ = 0;
  int32_t v_head_num_ = 0;
  int32_t page_size_ = 0;
  int32_t batch_size_ = 0;
};

extern "C" __global__ __aicore__ void reshape_and_cache_npd(GM_ADDR key, GM_ADDR value, GM_ADDR key_cache,
                                                            GM_ADDR value_cache, GM_ADDR slot_mapping, GM_ADDR q_seq,
                                                            GM_ADDR kv_seq, GM_ADDR block_tbl, GM_ADDR k_out,
                                                            GM_ADDR v_out, GM_ADDR workspace, GM_ADDR tiling) {
  if (TILING_KEY_IS(0)) {
    ReshapeAndCache<false, false, half> r;
    r.Init(key, value, key_cache, value_cache, slot_mapping, q_seq, kv_seq, block_tbl, k_out, v_out, workspace, tiling);
    r.Process();
  } else if (TILING_KEY_IS(1)) {
    ReshapeAndCache<true, false, half> r;
    r.Init(key, value, key_cache, value_cache, slot_mapping, q_seq, kv_seq, block_tbl, k_out, v_out, workspace, tiling);
    r.Process();
  } else if (TILING_KEY_IS(2)) {
    ReshapeAndCache<false, true, half> r;
    r.Init(key, value, key_cache, value_cache, slot_mapping, q_seq, kv_seq, block_tbl, k_out, v_out, workspace, tiling);
    r.Process();
  } else if (TILING_KEY_IS(3)) {
    ReshapeAndCache<true, true, half> r;
    r.Init(key, value, key_cache, value_cache, slot_mapping, q_seq, kv_seq, block_tbl, k_out, v_out, workspace, tiling);
    r.Process();
  }
}

#endif
