# eval.py
from random import sample

import torch
import torch.nn.functional as F
import difflib
from typing import List, Dict, Any, Tuple
from transformers import BertTokenizerFast
import numpy as np
from torch.utils.data import DataLoader

from Model import HateSpeechDetectionModel
from Hype import *

# BIO Tag Definitions (should match Model.py)
O_TAG = 0
B_T_TAG = 1  # Begin-Target
I_T_TAG = 2  # Inside-Target
B_A_TAG = 3  # Begin-Argument
I_A_TAG = 4  # Inside-Argument
NUM_BIO_TAGS = 5 # Should match Model.NUM_BIO_TAGS if accessed like that

NMS_IOU_THRESHOLD = 0.7 # Threshold for NMS

test_idx = 0


# --- Helper Function for IoU Calculation ---
def calculate_iou(span1_start: int, span1_end: int, span2_start: int, span2_end: int) -> float:
    """Calculates Intersection over Union for two token spans."""
    intersection_start = max(span1_start, span2_start)
    intersection_end = min(span1_end, span2_end)
    intersection_length = max(0, intersection_end - intersection_start + 1)

    if intersection_length == 0:
        return 0.0

    span1_length = span1_end - span1_start + 1
    span2_length = span2_end - span2_start + 1
    union_length = span1_length + span2_length - intersection_length

    if union_length == 0:
        return 0.0 # Should not happen if intersection_length > 0

    return intersection_length / union_length


# --- Helper Functions for Soft Match F1 Calculation ---
# 移到顶部，方便 _is_soft_match 和 calculate_f1_scores 调用
def _parse_quad_string(quad_str: str) -> Dict[str, str]:
    """解析四元组字符串为字典"""
    # 修正：使用 ' | ' 作为分隔符，确保与生成时的格式一致
    parts = [p.strip() for p in quad_str.split(' | ')]
    if len(parts) == 4:
        return {
            'target': parts[0],
            'argument': parts[1],
            'group': parts[2],
            'hateful': parts[3]
        }
    # 否则，可能是格式不正确，返回一个能被后续处理的默认值
    return {'target': '', 'argument': '', 'group': '', 'hateful': ''}


def _is_soft_match(pred_quad: Dict[str, str], true_quad: Dict[str, str], similarity_threshold: float = 0.5) -> bool:
    """
    判断两个四元组是否构成软匹配。
    条件：Targeted Group 和 Hateful 完全一致，Target 和 Argument 相似度 > 阈值。
    """
    # Targeted Group 和 Hateful 必须完全一致
    if pred_quad['group'] != true_quad['group'] or \
            pred_quad['hateful'] != true_quad['hateful']:
        return False

    # 计算 Target 和 Argument 的字符串相似度
    target_sim = difflib.SequenceMatcher(None, pred_quad['target'], true_quad['target']).ratio()
    argument_sim = difflib.SequenceMatcher(None, pred_quad['argument'], true_quad['argument']).ratio()

    return target_sim >= similarity_threshold and argument_sim >= similarity_threshold


# --- 主要评估函数 ---
DETAILED_LOG_SAMPLES = 5 # Number of samples to log in detail per evaluation run

def evaluate_model(model: HateSpeechDetectionModel,
                   val_dataloader: DataLoader,
                   tokenizer: BertTokenizerFast,
                   device: torch.device) -> Dict[str, float]:
    """
    在验证集上评估模型性能，计算硬匹配和软匹配 F1 分数。
    """
    global test_idx
    # Reset test_idx at the beginning of each evaluation call if we want DETAILED_LOG_SAMPLES per run
    # test_idx = 0 # Optional: uncomment if test_idx should reset per evaluate_model call

    model.eval()  # 设置模型为评估模式
    all_predicted_quads = []  # 存储所有样本的预测四元组列表
    all_true_quads = []  # 存储所有样本的真实四元组列表

    with torch.no_grad():  # 禁用梯度计算
        for batch_idx, batch in enumerate(val_dataloader):
            input_ids = batch['input_ids'].to(device)  # [B, L]
            attention_mask = batch['attention_mask'].to(device)  # [B, L]
            token_type_ids = batch['token_type_ids'].to(device)
            # original_contents = batch['original_content'] # 如果需要原始char-level内容，从DataLoader获取
            true_quads_batch = batch['quads_labels']  # 真实的 token-level quads 标签

            # 1. 模型前向传播获取原始 logits
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

            batch_size, seq_len = input_ids.size()

            for i in range(batch_size):  # 遍历 batch 中的每个样本
                current_input_ids_sample = input_ids[i]
                current_attention_mask_sample = attention_mask[i]
                current_true_quads_sample = true_quads_batch[i]
                real_len_sample = int(current_attention_mask_sample.sum().item())

                # Determine if detailed logging is active for this sample
                # test_idx is incremented later, so we check its current value before increment
                # Let's adjust the detailed log condition to be based on a counter specific to this eval run
                # or manage test_idx carefully.
                # For simplicity, let's use a condition that logs first few samples of first batch.
                # This will be less chatty than using global test_idx directly if it's not reset.
                # However, the prompt mentions using existing test_idx.
                # The existing logic is: test_idx +=1; if test_idx >= 80: print; test_idx = 0
                # So, if we want to log for test_idx 1 to 5 (after increment):
                # We'd check test_idx (before increment) == 0 up to 4.
                # Let's use a simpler check: log if current sample_idx_in_batch < DETAILED_LOG_SAMPLES and batch_idx == 0
                # Or, stick to global test_idx:
                # The prompt says "Use the existing test_idx global variable ... to log details for a limited number of samples"
                # "test_idx is incremented later" - this is key.

                # New detailed logging block
                # We will use the global test_idx. The condition `test_idx < DETAILED_LOG_SAMPLES`
                # will apply *after* test_idx is incremented later in the loop.
                # So, to log for the *first* DETAILED_LOG_SAMPLES, we'd want to check test_idx's value
                # *before* it's incremented for the current sample.
                # Let's make detailed_log_active decision *before* incrementing test_idx.
                # The current structure increments test_idx *after* processing the sample.
                # So, if test_idx is 0, this is the 1st sample to be processed by this eval_model call (if test_idx is reset)

                # For this subtask, let's assume test_idx is reset to 0 at the start of `evaluate_model`
                # or that we only care about the first few samples globally until it resets.
                # The prompt says "test_idx += 1" and then "if test_idx >= 80 ... test_idx = 0"
                # This means test_idx effectively cycles.
                # Let's log if test_idx is in the range [0, DETAILED_LOG_SAMPLES - 1] for the *upcoming* sample.

                detailed_log_active = (test_idx < DETAILED_LOG_SAMPLES)

                if detailed_log_active:
                    print(f"\n---- Eval Span Analysis for Sample (test_idx: {test_idx}) ----")
                    # Log input text
                    decoded_input_text = tokenizer.decode(current_input_ids_sample[:real_len_sample], skip_special_tokens=True)
                    print(f"Input Text: \"{decoded_input_text}\"")

                    # Log gold Target and Argument spans
                    gold_target_spans_tokens = []
                    gold_target_spans_text = []
                    gold_argument_spans_tokens = []
                    gold_argument_spans_text = []

                    for quad in current_true_quads_sample:
                        ts, te = quad['t_start_token'].item(), quad['t_end_token'].item()
                        as_, ae = quad['a_start_token'].item(), quad['a_end_token'].item()

                        if ts != -1 and te != -1 and 0 <= ts <= te < real_len_sample:
                            gold_target_spans_tokens.append((ts, te))
                            gold_target_spans_text.append(tokenizer.decode(current_input_ids_sample[ts:te+1], skip_special_tokens=True).replace(" ", ""))

                        if as_ != -1 and ae != -1 and 0 <= as_ <= ae < real_len_sample:
                            gold_argument_spans_tokens.append((as_, ae))
                            gold_argument_spans_text.append(tokenizer.decode(current_input_ids_sample[as_:ae+1], skip_special_tokens=True).replace(" ", ""))

                    print(f"Gold Target Spans (Tokens): {gold_target_spans_tokens}")
                    print(f"Gold Target Spans (Text): {gold_target_spans_text}")
                    print(f"Gold Argument Spans (Tokens): {gold_argument_spans_tokens}")
                    print(f"Gold Argument Spans (Text): {gold_argument_spans_text}")

                    # Log BIO tag probabilities (first 30 tokens)
                    num_tokens_to_log = min(30, real_len_sample)
                    if 'bio_tag_logits' in outputs:
                        bio_probs_sample = torch.softmax(outputs['bio_tag_logits'][i][:num_tokens_to_log], dim=-1).tolist()
                        print(f"BIO Tag Probs (Sample {test_idx}, first {num_tokens_to_log} tokens): {bio_probs_sample}")
                    else:
                        print(f"BIO Tag Logits not found in model outputs for Sample {test_idx}.")
                    # End of new detailed logging block

                # 从 batch outputs 中提取当前样本的 outputs for convert_logits_to_quads
                # This now needs to pass 'bio_tag_logits' instead of start/end logits
                sample_outputs_for_quad_conversion = {
                    'bio_tag_logits': outputs['bio_tag_logits'][i].unsqueeze(0), # Pass BIO logits
                    'sequence_output': outputs['sequence_output'][i].unsqueeze(0),
                    'cls_output': outputs['cls_output'][i].unsqueeze(0),
                    # 'input_ids': input_ids[i].unsqueeze(0), # Already available as sample_input_ids
                    'attention_mask': attention_mask[i].unsqueeze(0)
                }

                # print("t_s logits:", sample_outputs_for_quad_conversion['target_start_logits'].squeeze(1))
                # print("t_e logits:", sample_outputs_for_quad_conversion['target_end_logits'])
                # print("a_s logits:", sample_outputs_for_quad_conversion['argument_start_logits'])
                # print("a_e logits:", sample_outputs_for_quad_conversion['argument_end_logits'])

                sample_input_ids = input_ids[i].unsqueeze(0)  # [1, L]

                # 将当前样本的原始预测 logits 转换为四元组字符串列表
                predicted_quads_for_sample = convert_logits_to_quads(
                    outputs_for_a_sample=sample_outputs_for_quad_conversion,
                    sample_input_ids=sample_input_ids,
                    tokenizer=tokenizer,
                    max_seq_length=MAX_SEQ_LENGTH, # This is seq_len from outputs, not Hype.MAX_SEQ_LENGTH
                    model_instance=model,
                    detailed_log_active=detailed_log_active, # Pass the flag
                    test_idx_for_log=test_idx # Pass current test_idx for consistent logging
                )
                all_predicted_quads.append(predicted_quads_for_sample)

                # 构造当前样本的“真实四元组字符串列表”
                # This part uses current_input_ids_sample, current_attention_mask_sample, real_len_sample, current_true_quads_sample
                true_quads_for_sample = []
                ids_i = current_input_ids_sample # Use current sample's input_ids
                # real_len is already real_len_sample

                for quad in current_true_quads_sample: # Use current sample's true quads
                    ts = quad['t_start_token'].item()
                    te = quad['t_end_token'].item()
                    as_ = quad['a_start_token'].item()
                    ae = quad['a_end_token'].item()

                    target_text = ""
                    argument_text = ""

                    # 修正：Span 索引范围检查，确保在真实文本范围内
                    if ts != -1 and te != -1 and 0 <= ts <= te < real_len:  # 使用 real_len
                        target_text = tokenizer.decode(
                            ids_i[ts:te + 1], skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )

                        target_text = target_text.replace(" ", "") if target_text.replace(" ", "") != "" else "NULL"
                    else:
                        target_text = "NULL"

                    if as_ != -1 and ae != -1 and 0 <= as_ <= ae < real_len:  # 使用 real_len
                        argument_text = tokenizer.decode(
                            ids_i[as_:ae + 1], skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )

                        argument_text = argument_text.replace(" ", "") if argument_text.replace(" ",
                                                                                                "") != "" else "NULL"
                    else:
                        argument_text = "NULL"

                    # 修正：`NULL` 字符串处理，使用原始 `target_text_raw` / `argument_text_raw`
                    # 这依赖于 data_loader.py 将这两个字段传递过来
                    if quad['target_text_raw'] == 'NULL':
                        target_text = 'NULL'
                    elif not target_text and quad['target_text_raw']:  # 如果解码为空，但原始标签不为NULL，则用原始标签文本
                        target_text = quad['target_text_raw']

                    if quad['argument_text_raw'] == 'NULL':
                        argument_text = 'NULL'
                    elif not argument_text and quad['argument_text_raw']:
                        argument_text = quad['argument_text_raw']

                    # 修正：Group 标签的标准化拼接 (已在 loss.py 中确保排序，这里再次保证)
                    group_labels_list = [TARGET_GROUP_CLASS_NAME[idx]
                                         for idx, val in enumerate(quad['group_vector'].tolist()) if val == 1]
                    group_labels_list.sort(key=lambda x: TARGET_GROUP_CLASS_NAME.index(x))  # 确保按Hype中定义的顺序
                    group_str = ','.join(group_labels_list) if group_labels_list else 'non-hate'

                    hateful_str = 'hate' if quad['hateful_flag'].item() == 1 else 'non-hate'

                    # 修正：字符串拼接格式统一，使用 ' | '
                    true_quads_for_sample.append(f"{target_text} | {argument_text} | {group_str} | {hateful_str}")

                # Existing logging based on test_idx cycling up to 80
                # This detailed_log_active is for the *new* span analysis logging.
                # The existing print condition for predicted/true quads should remain.
                # The increment of test_idx should happen once per sample.

                if detailed_log_active or (test_idx >= 79 and test_idx < 80 + DETAILED_LOG_SAMPLES -1): # Adjust existing log for clarity or keep as is
                     # The original logic was `test_idx +=1; if test_idx >= 80: print; test_idx = 0`
                     # This means it prints when test_idx becomes 80.
                     # Let's keep the original logic for this part to minimize unintended changes.
                     pass # New logging is already done. This comment is for reasoning.

                test_idx += 1 # Increment for each sample processed in the batch
                if test_idx >= 80: # Original condition for printing summarized quads
                    print(f"\n--- Summarized Quads for Sample (global test_idx reset point: {test_idx}) ---")
                    print("Predicted Quads for sample:", predicted_quads_for_sample)
                    print("True Quads for sample:", true_quads_for_sample)
                    test_idx = 0 # Reset global test_idx

                all_true_quads.append(true_quads_for_sample)

    metrics = calculate_f1_scores(all_predicted_quads, all_true_quads)
    return metrics


# -------------- convert_logits_to_quads --------------
def convert_logits_to_quads(outputs_for_a_sample: Dict[str, torch.Tensor],
                            sample_input_ids: torch.Tensor,  # 单样本的 input_ids
                            tokenizer: BertTokenizerFast,
                            max_seq_length: int,  # 实际上 seq_len 已经从 logits 中获取
                            model_instance: HateSpeechDetectionModel,
                            detailed_log_active: bool, # New flag
                            test_idx_for_log: int # New var for log header
                            ) -> List[str]:
    """
    将单个样本的 logits 转换为预测四元组（字符串形式）
    """
    predicted_quads_strings = []
    # The ids for decoding text for top-k spans
    ids_for_decoding = sample_input_ids.squeeze(0) # Shape: [L]

    bio_tag_logits = outputs_for_a_sample['bio_tag_logits'].squeeze(0) # Shape: (seq_len, NUM_BIO_TAGS)
    sequence_output = outputs_for_a_sample['sequence_output']  # [1, L, H]
    cls_output = outputs_for_a_sample['cls_output']  # [1, H]
    attention_mask = outputs_for_a_sample['attention_mask'].squeeze(0)  # [L]

    seq_len = bio_tag_logits.size(0)
    real_len = int(attention_mask.sum().item())
    valid_span_start_idx = 1  # Skip [CLS]
    valid_span_end_idx = real_len - 2  # Skip [SEP] (and padding)

    if valid_span_end_idx < valid_span_start_idx:
        return []

    predicted_bio_probs = torch.softmax(bio_tag_logits, dim=-1) # Shape: (seq_len, NUM_BIO_TAGS)
    predicted_bio_tags = torch.argmax(predicted_bio_probs, dim=-1) # Shape: (seq_len)

    if detailed_log_active:
        print(f"Predicted BIO Tags (Sample {test_idx_for_log}, up to real_len): {predicted_bio_tags[:real_len].tolist()}")

    # Helper function to extract spans from BIO tags
    def extract_spans_from_bio_tags(tags: torch.Tensor, probs: torch.Tensor,
                                    current_real_len: int, b_tag_val: int, i_tag_val: int,
                                    start_offset: int, end_offset: int) -> List[Tuple[int, int, float]]:
        spans = []
        idx = start_offset
        while idx <= end_offset:
            if tags[idx].item() == b_tag_val:
                span_start = idx
                span_end = idx
                span_prob_product = probs[idx, tags[idx]].item()

                # Try to extend with I-tags
                next_idx = idx + 1
                while next_idx <= end_offset and tags[next_idx].item() == i_tag_val:
                    span_prob_product *= probs[next_idx, tags[next_idx]].item()
                    span_end = next_idx
                    next_idx += 1

                # Score: product of probabilities of the tags in the span
                # Add 1e-9 for numerical stability if product is zero, though should not happen with softmax
                score = span_prob_product
                spans.append((span_start, span_end, score))
                idx = next_idx # Continue search from where the current span ended
            else:
                idx += 1
        return spans

    candidate_target_spans = extract_spans_from_bio_tags(
        predicted_bio_tags, predicted_bio_probs, real_len, B_T_TAG, I_T_TAG,
        valid_span_start_idx, valid_span_end_idx
    )
    candidate_argument_spans = extract_spans_from_bio_tags(
        predicted_bio_tags, predicted_bio_probs, real_len, B_A_TAG, I_A_TAG,
        valid_span_start_idx, valid_span_end_idx
    )

    candidate_target_spans.sort(key=lambda x: x[2], reverse=True)
    candidate_argument_spans.sort(key=lambda x: x[2], reverse=True)

    if detailed_log_active:
        print(f"---- Top K BIO-Extracted Spans in convert_logits_to_quads for Sample (test_idx: {test_idx_for_log}) ----")
        logged_target_spans = []
        for ts, te, score in candidate_target_spans[:3]: # Log top 3
            if 0 <= ts < ids_for_decoding.size(0) and 0 <= te < ids_for_decoding.size(0) and ts <= te:
                text = tokenizer.decode(ids_for_decoding[ts:te+1], skip_special_tokens=True).replace(" ", "")
                logged_target_spans.append( ((ts, te), text, f"{score:.4f}") )
            else:
                logged_target_spans.append( ((ts, te), "[INVALID INDICES]", f"{score:.4f}") )
        print(f"Top {len(logged_target_spans)} Predicted Target Spans (BIO): {logged_target_spans}")

        logged_argument_spans = []
        for as_idx, ae_idx, score in candidate_argument_spans[:3]: # Log top 3
            if 0 <= as_idx < ids_for_decoding.size(0) and 0 <= ae_idx < ids_for_decoding.size(0) and as_idx <= ae_idx:
                text = tokenizer.decode(ids_for_decoding[as_idx:ae_idx+1], skip_special_tokens=True).replace(" ", "")
                logged_argument_spans.append( ((as_idx, ae_idx), text, f"{score:.4f}") )
            else:
                logged_argument_spans.append( ((as_idx, ae_idx), "[INVALID INDICES]", f"{score:.4f}") )
        print(f"Top {len(logged_argument_spans)} Predicted Argument Spans (BIO): {logged_argument_spans}")

    # 2. Biaffine 配对
    final_quad_candidates = []

    if not candidate_target_spans or not candidate_argument_spans:
        return []

    target_vecs = []
    target_idx_map = []
    for (ts, te, _) in candidate_target_spans:
        vec = model_instance._get_span_representation(sequence_output, ts, te, batch_idx=0)  # sequence_output已是单样本
        target_vecs.append(vec)
        target_idx_map.append((ts, te))

    argument_vecs = []
    argument_idx_map = []
    for (as_idx, ae_idx, _) in candidate_argument_spans:
        vec = model_instance._get_span_representation(sequence_output, as_idx, ae_idx, batch_idx=0)
        argument_vecs.append(vec)
        argument_idx_map.append((as_idx, ae_idx))

    if not target_vecs or not argument_vecs:
        return []

    target_mat = torch.stack(target_vecs, dim=0)  # [N_t, H]
    argument_mat = torch.stack(argument_vecs, dim=0)  # [N_a, H]

    pair_logits = model_instance.biaffine_pairing(target_mat, argument_mat)  # [N_t, N_a]
    pair_probs = torch.sigmoid(pair_logits)
    # print("pair_probs:", pair_probs)

    # 3. NMS-based Filtering and Selection
    N_t_cand, N_a_cand = pair_probs.size()

    threshold_passed_pairs = []
    for i_t in range(N_t_cand):
        for j_a in range(N_a_cand):
            biaffine_score = pair_probs[i_t, j_a].item()
            if biaffine_score > PAIRING_THRESHOLD:
                # candidate_target_spans[i_t] is (ts, te, bio_score)
                # candidate_argument_spans[j_a] is (as, ae, bio_score)
                threshold_passed_pairs.append({
                    't_span_data': candidate_target_spans[i_t], # (ts, te, bio_score)
                    'a_span_data': candidate_argument_spans[j_a], # (as, ae, bio_score)
                    'pair_score': biaffine_score
                })

    # Sort pairs by Biaffine score (descending)
    threshold_passed_pairs.sort(key=lambda x: x['pair_score'], reverse=True)

    selected_quad_candidates_after_nms = []
    for cand_pair in threshold_passed_pairs:
        current_t_span = cand_pair['t_span_data'] # (ts, te, t_bio_score)
        current_a_span = cand_pair['a_span_data'] # (as, ae, a_bio_score)

        is_redundant = False
        for selected_item in selected_quad_candidates_after_nms:
            prev_t_span = selected_item['target_span_details'] # This was how it was stored in previous version
            prev_a_span = selected_item['argument_span_details']

            target_iou = calculate_iou(current_t_span[0], current_t_span[1], prev_t_span[0], prev_t_span[1])
            argument_iou = calculate_iou(current_a_span[0], current_a_span[1], prev_a_span[0], prev_a_span[1])

            if target_iou > NMS_IOU_THRESHOLD and argument_iou > NMS_IOU_THRESHOLD:
                is_redundant = True
                break

        if not is_redundant:
            selected_quad_candidates_after_nms.append({
                't_start_token': current_t_span[0],
                't_end_token': current_t_span[1],
                'a_start_token': current_a_span[0],
                'a_end_token': current_a_span[1],
                'pair_score': cand_pair['pair_score'],
                'target_span_details': current_t_span, # For NMS check with subsequent items
                'argument_span_details': current_a_span # For NMS check with subsequent items
            })

    final_quad_candidates = selected_quad_candidates_after_nms

    # Fallback: If NMS (and thresholding) yields no candidates, consider taking the single best pre-NMS pair.
    # This is a simplified fallback. A more robust one might re-run NMS on top-K pre-threshold pairs.
    if not final_quad_candidates and candidate_pairs_scores: # candidate_pairs_scores is from before any filtering
         # candidate_pairs_scores was: list of (score, i_t, j_a), sorted by score
         # This part is tricky because candidate_pairs_scores was defined in the old version.
         # Let's rebuild a simplified version of candidate_pairs_scores for the fallback.
         all_possible_pairs_for_fallback = []
         if not candidate_target_spans or not candidate_argument_spans: # Should have been caught earlier
             pass # No spans to form pairs from
         else:
             for i_t_fallback in range(len(candidate_target_spans)):
                 for j_a_fallback in range(len(candidate_argument_spans)):
                     score_fallback = pair_probs[i_t_fallback, j_a_fallback].item()
                     all_possible_pairs_for_fallback.append({
                         't_span_data': candidate_target_spans[i_t_fallback],
                         'a_span_data': candidate_argument_spans[j_a_fallback],
                         'pair_score': score_fallback
                     })
             all_possible_pairs_for_fallback.sort(key=lambda x: x['pair_score'], reverse=True)

             if all_possible_pairs_for_fallback: # If there are any pairs at all
                top_fallback_pair = all_possible_pairs_for_fallback[0]
                final_quad_candidates.append({
                    't_start_token': top_fallback_pair['t_span_data'][0],
                    't_end_token': top_fallback_pair['t_span_data'][1],
                    'a_start_token': top_fallback_pair['a_span_data'][0],
                    'a_end_token': top_fallback_pair['a_span_data'][1],
                    'pair_score': top_fallback_pair['pair_score']
                    # Not storing span_details here as this is a simple fallback append
                })


    # Classification and string formatting part remains largely the same,
    # but now it operates on `final_quad_candidates` which has undergone NMS.
    # The MAX_PREDICTED_QUADS_PER_SAMPLE is applied *after* classification and re-sorting.
    # This means NMS reduces candidates, then classification happens, then a final top-K cut.

    # 4. 分类并输出成字符串
    ids = sample_input_ids.squeeze(0)  # [L]

    if not final_quad_candidates:
        return []  # 如果没有预测出任何四元组，直接返回空列表

    all_quad_reprs = []
    for quad_cand in final_quad_candidates:
        ts = quad_cand['t_start_token']
        te = quad_cand['t_end_token']
        as_idx = quad_cand['a_start_token']
        ae_idx = quad_cand['a_end_token']

        # 从单样本的 sequence_output 中获取 Span 表示
        target_vec = model_instance._get_span_representation(sequence_output, ts, te, 0)
        argument_vec = model_instance._get_span_representation(sequence_output, as_idx, ae_idx, 0)

        combined_vec = torch.cat([target_vec, argument_vec, cls_output.squeeze(0)], dim=-1)  # cls_output 已经是 (1, H)
        all_quad_reprs.append(combined_vec)

    stacked_quad_reprs = torch.stack(all_quad_reprs, dim=0)  # (num_predicted_quads, 3*hidden_size)

    # 批量调用分类器
    group_logits_batch = model_instance.group_classifier(stacked_quad_reprs)
    hateful_logits_batch = model_instance.hateful_classifier(stacked_quad_reprs)

    group_probs_batch = torch.sigmoid(group_logits_batch)
    hateful_probs_batch = torch.sigmoid(hateful_logits_batch)

    # 遍历预测结果并格式化
    for idx, quad_cand in enumerate(final_quad_candidates):
        ts = quad_cand['t_start_token']
        te = quad_cand['t_end_token']
        as_idx = quad_cand['a_start_token']
        ae_idx = quad_cand['a_end_token']

        group_probs_single = group_probs_batch[idx]  # (num_groups)
        hateful_prob_single = hateful_probs_batch[idx].item()  # scalar

        # 确定 Targeted Group (多标签)
        predicted_groups = []
        for i, prob in enumerate(group_probs_single):
            if prob.item() > 0.5:  # 可以调整阈值
                predicted_groups.append(TARGET_GROUP_CLASS_NAME[i])

        # 修正：保证顺序一致，并使用','分隔 (已在 calculate_f1_scores 的 helper 中处理)
        predicted_groups.sort(key=lambda x: TARGET_GROUP_CLASS_NAME.index(x))  # 确保按Hype中定义的顺序
        group_str = ','.join(predicted_groups) if predicted_groups else 'non-hate'

        # 确定 Hateful
        hateful_str = 'hate' if hateful_prob_single > 0.5 else 'non-hate'

        # 将 token span 转换为文本 (使用 sample_input_ids)
        target_text = ""
        argument_text = ""

        # 确保 ts, te, as_idx, ae_idx 在 token 序列的有效范围内
        # 这里的 seq_len 是原始的 token 序列长度，包含 padding
        if ts != -1 and te != -1 and 0 <= ts <= te < seq_len:
            target_ids = ids[ts:te + 1].tolist()
            target_text = tokenizer.decode(target_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            target_text = target_text.replace(" ", "") if target_text.replace(" ", "") != "" else "NULL"
        else:
            target_text = 'NULL'

        if as_idx != -1 and ae_idx != -1 and 0 <= as_idx <= ae_idx < seq_len:
            argument_ids = ids[as_idx:ae_idx + 1].tolist()
            argument_text = tokenizer.decode(argument_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            argument_text = argument_text.replace(" ", "") if argument_text.replace(" ", "") != "" else "NULL"
        else:
            argument_text = "NULL"
        # 修正：对于 'NULL' 字符串的特殊处理
        # 如果 `decode` 结果是空字符串，就将其替换为 'NULL'。
        target_text = target_text if target_text.strip() else 'NULL'  # 使用 .strip() 处理空字符串和纯空白字符串
        argument_text = argument_text if argument_text.strip() else 'NULL'

        # 格式化为最终输出字符串
        # 修正：字符串拼接格式统一，使用 ' | '
        predicted_quads_strings.append(f"{target_text} | {argument_text} | {group_str} | {hateful_str}")
    # print(predicted_quads_strings)
    return predicted_quads_strings


# --------------- calculate_f1_scores ---------------
def calculate_f1_scores(pred_quads_list: List[List[str]], true_quads_list: List[List[str]]) -> Dict[str, float]:
    """
    计算硬匹配和软匹配 F1 分数。
    pred_quads_list: List[List[str]]，每个内层列表是单个样本预测的四元组字符串。
    true_quads_list: List[List[str]]，每个内层列表是单个样本真实的四元组字符串。
    """
    tp_hard, fp_hard, fn_hard = 0, 0, 0
    tp_soft, fp_soft, fn_soft = 0, 0, 0

    assert len(pred_quads_list) == len(true_quads_list)

    for preds_for_sample, truths_for_sample in zip(pred_quads_list, true_quads_list):
        # 修正：如果预测和真实都为空，则不影响统计 (TP=0, FP=0, FN=0)
        if not preds_for_sample and not truths_for_sample:
            continue

        # 硬匹配 (Exact Match)
        matched_preds_hard = [False] * len(preds_for_sample)
        matched_truths_hard = [False] * len(truths_for_sample)

        for p_idx, pred_q_str in enumerate(preds_for_sample):
            for t_idx, true_q_str in enumerate(truths_for_sample):
                if pred_q_str == true_q_str and not matched_truths_hard[t_idx]:
                    tp_hard += 1
                    matched_preds_hard[p_idx] = True
                    matched_truths_hard[t_idx] = True
                    break  # 一个预测最多匹配一个真实

        fp_hard += sum(1 for matched in matched_preds_hard if not matched)
        fn_hard += sum(1 for matched in matched_truths_hard if not matched)

        # 软匹配 (Partial Match)
        # 修正：_parse_quad_string 已经处理了 ' | ' 分隔符
        parsed_preds = [_parse_quad_string(x) for x in preds_for_sample]
        parsed_truths = [_parse_quad_string(x) for x in truths_for_sample]

        matched_preds_soft = [False] * len(parsed_preds)
        matched_truths_soft = [False] * len(parsed_truths)

        for p_idx, pred_quad in enumerate(parsed_preds):
            for t_idx, true_quad in enumerate(parsed_truths):
                if not matched_truths_soft[t_idx] and _is_soft_match(pred_quad, true_quad):
                    tp_soft += 1
                    matched_preds_soft[p_idx] = True
                    matched_truths_soft[t_idx] = True
                    break

        fp_soft += sum(1 for m in matched_preds_soft if not m)
        fn_soft += sum(1 for m in matched_truths_soft if not m)

    # 计算 F1 指标
    hard_precision = tp_hard / (tp_hard + fp_hard) if (tp_hard + fp_hard) > 0 else 0.0
    hard_recall = tp_hard / (tp_hard + fn_hard) if (tp_hard + fn_hard) > 0 else 0.0
    hard_f1 = (2 * hard_precision * hard_recall) / (hard_precision + hard_recall) \
        if (hard_precision + hard_recall) > 0 else 0.0

    soft_precision = tp_soft / (tp_soft + fp_soft) if (tp_soft + fp_soft) > 0 else 0.0
    soft_recall = tp_soft / (tp_soft + fn_soft) if (tp_soft + fn_soft) > 0 else 0.0
    soft_f1 = (2 * soft_precision * soft_recall) / (soft_precision + soft_recall) \
        if (soft_precision + soft_recall) > 0 else 0.0

    avg_f1 = (hard_f1 + soft_f1) / 2.0

    return {
        "hard_precision": hard_precision,
        "hard_recall": hard_recall,
        "hard_f1": hard_f1,
        "soft_precision": soft_precision,
        "soft_recall": soft_recall,
        "soft_f1": soft_f1,
        "average_f1": avg_f1
    }


# --- 辅助函数 ---
# 移到顶部，避免重复定义

# --- 快速测试 calculate_f1_scores ---
if __name__ == '__main__':
    print("--- Testing calculate_f1_scores ---")

    # 示例数据（调整以匹配 ' | ' 分隔符）
    pred1 = ["t1 | a1 | Racism,Sexism | hate"]
    true1 = ["t1 | a1 | Racism | hate"]  # Group不完全匹配，硬匹配失败，软匹配也失败

    pred2 = ["A | B | Sexism | non-hate"]
    true2 = ["A | B | Sexism | non-hate", "C | D_wrong | Sexism | non-hate"]

    pred3 = ["CCC | DDD | LGBTQ | hate"]
    true3 = ["C | D | LGBTQ | hate"]  # 软匹配成功

    pred4 = ["X | Y | Racism | non-hate"]
    true4 = []  # 预测有，真实无

    pred5 = []
    true5 = ["Z | W | Sexism | hate"]  # 预测无，真实有

    pred6 = []
    true6 = []  # 预测无，真实无

    all_predicted = [pred1, pred2, pred3, pred4, pred5, pred6]
    all_true = [true1, true2, true3, true4, true5, true6]
    metrics = calculate_f1_scores(all_predicted, all_true)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")