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
void __aicore__ __inline__ reshape_and_cache_npd_kv_npd_offset(int token_idx, int block_size, int head_num, int embed,
                                                               __gm__ uint8_t *__restrict__ kv_seq,
                                                               uint32_t *token_offset) {
  uint32_t offset = 0;
  uint32_t i = 0;
  int32_t limit = (int32_t)(*((__gm__ int32_t *)kv_seq + i));
  int32_t prev_seq_len = limit;

  uint32_t prev_token = 0;
  while (token_idx >= limit) {
    offset += ((prev_seq_len + block_size - 1) / block_size) * block_size;
    prev_token += prev_seq_len;
    i++;
    prev_seq_len = (int32_t)(*((__gm__ int32_t *)kv_seq + i));
    limit += prev_seq_len;
  }
  uint32_t inner_token_idx =  token_idx - prev_token;
  uint32_t batch_offset = offset * head_num * embed;
  uint32_t inner_batch_offset = (inner_token_idx / block_size) * head_num * block_size * embed +
                                (inner_token_idx % block_size) * embed;
  *token_offset = batch_offset + inner_batch_offset;
}

template <typename Dtype>
__aicore__ __inline__ bool reshape_and_cache_npd_kv_copy(__gm__ uint8_t *__restrict__ input_gm,
                                                         __gm__ uint8_t *__restrict__ slot_mapping_input_gm,
                                                         __gm__ uint8_t *__restrict__ q_seq,
                                                         __gm__ uint8_t *__restrict__ kv_seq,
                                                         __gm__ uint8_t *__restrict__ block_tbl,
                                                         int32_t block_size,
                                                         int32_t head_num,
                                                         int32_t embed,
                                                         int64_t start_index,
                                                        __gm__ uint8_t *__restrict__ cache_output_gm,
                                                        __gm__ uint8_t *__restrict__ update_output_gm) {
  int64_t hidden_size = head_num * embed;
  int64_t start = start_index * hidden_size;
  int32_t slot_value = (int32_t)(*((__gm__ int32_t *)slot_mapping_input_gm + start_index));
  if (slot_value < 0) return false;
  int32_t num_hidden_blocks = hidden_size * sizeof(Dtype) / BLOCK_SIZE;

  __ubuf__ uint8_t *temp_ubuf = (__ubuf__ uint8_t *)get_imm(0);  // 临时存放 token
  copy_gm_to_ubuf((__ubuf__ Dtype *)temp_ubuf, (__gm__ Dtype *)input_gm + start,
                  0,                  // sid
                  1,                  // nBurst
                  num_hidden_blocks,  // lenBurst
                  0,                  // srcGap
                  0);                 // dstGap
  set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);

  // cache is stored [B_NUM, N, B_SIZE, D]
  int64_t cache_start = (slot_value / block_size) * head_num * block_size * embed + (slot_value % block_size) * embed;
  int32_t dst_stride = (block_size - 1) * embed * sizeof(Dtype) / BLOCK_SIZE;
  int32_t num_embed_blocks = embed * sizeof(Dtype) / BLOCK_SIZE;

  copy_ubuf_to_gm((__gm__ Dtype *)cache_output_gm + cache_start, (__ubuf__ Dtype *)temp_ubuf,
                  0,                 // sid
                  head_num,          // nBurst
                  num_embed_blocks,  // lenBurst
                  0,                 // srcGap
                  dst_stride);       // dstGap

  // update key\value - [T/B_SIZE, N, B_SIZE, D]
  // Get Token base offset
  uint32_t token_offset = 0;
  reshape_and_cache_npd_kv_npd_offset(start_index, block_size, head_num, embed, kv_seq, &token_offset);
  copy_ubuf_to_gm((__gm__ Dtype *)update_output_gm + token_offset, (__ubuf__ Dtype *)temp_ubuf,
                  0,                 // sid
                  head_num,          // nBurst
                  num_embed_blocks,  // lenBurst
                  0,                 // srcGap
                  dst_stride);       // dstGap

  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  return true;
}

template <typename Dtype>
__aicore__ __inline__ void reshape_and_cache_npd_kernel(
  __gm__ uint8_t *__restrict__ key_input_gm, __gm__ uint8_t *__restrict__ value_input_gm,
  __gm__ uint8_t *__restrict__ key_cache_gm, __gm__ uint8_t *__restrict__ value_cache_gm,
  __gm__ uint8_t *__restrict__ slot_mapping_input_gm, __gm__ uint8_t *__restrict__ q_seq,
  __gm__ uint8_t *__restrict__ kv_seq, __gm__ uint8_t *__restrict__ block_tbl, __gm__ uint8_t *__restrict__ k_out,
  __gm__ uint8_t *__restrict__ v_out, __gm__ uint8_t *__restrict__ workspace,
  __gm__ uint8_t *__restrict__ tiling_para_gm) {
  int32_t num_tokens = (int32_t)(*((__gm__ int32_t *)tiling_para_gm + 2));
  int32_t hidden_size = (int32_t)(*((__gm__ int32_t *)tiling_para_gm + 3));
  int32_t block_size = (int32_t)(*((__gm__ int32_t *)tiling_para_gm + 6));
  int32_t k_head_num = (int32_t)(*((__gm__ int32_t *)tiling_para_gm + 4));
  int32_t v_head_num = (int32_t)(*((__gm__ int32_t *)tiling_para_gm + 5));

  int32_t core_num = get_block_num();
  int64_t per_core_task_num = num_tokens * 2 / core_num;
  int32_t tail_task_num = num_tokens * 2 % core_num;

  int32_t block_id = get_block_idx();
  int64_t start_task_id = block_id * per_core_task_num;

  if (block_id < tail_task_num) {
    per_core_task_num++;
    start_task_id += block_id;
  } else {
    start_task_id += tail_task_num;
  }

  for (int64_t i = 0; i < per_core_task_num; i++) {
    if (i + start_task_id < num_tokens) {
      reshape_and_cache_npd_kv_copy<Dtype>(key_input_gm, slot_mapping_input_gm, q_seq, kv_seq, block_tbl, block_size,
                                           k_head_num, hidden_size / k_head_num, (i + start_task_id), key_cache_gm,
                                           k_out);
    } else {
      reshape_and_cache_npd_kv_copy<Dtype>(value_input_gm, slot_mapping_input_gm, q_seq, kv_seq, block_tbl, block_size,
                                           v_head_num, hidden_size / v_head_num, (i + start_task_id - num_tokens),
                                           value_cache_gm, v_out);
    }
  }
}

extern "C" __global__ __aicore__ void reshape_and_cache_npd(GM_ADDR key, GM_ADDR value, GM_ADDR key_cache,
                                                            GM_ADDR value_cache, GM_ADDR slot_mapping, GM_ADDR q_seq,
                                                            GM_ADDR kv_seq, GM_ADDR block_tbl, GM_ADDR k_out,
                                                            GM_ADDR v_out, GM_ADDR workspace, GM_ADDR tiling) {
  int32_t tiling_id = (int32_t)(*((__gm__ int32_t *)tiling + 0));
  if (tiling_id == 0) {
    reshape_and_cache_npd_kernel<half>(key, value, key_cache, value_cache, slot_mapping, q_seq, kv_seq, block_tbl,
                                       k_out, v_out, workspace, tiling);
  }
}

#endif
