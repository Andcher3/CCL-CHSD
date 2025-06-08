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

test_idx = 0


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
def evaluate_model(model: HateSpeechDetectionModel,
                   val_dataloader: DataLoader,
                   tokenizer: BertTokenizerFast,
                   device: torch.device) -> Dict[str, float]:
    """
    在验证集上评估模型性能，计算硬匹配和软匹配 F1 分数。
    """
    global test_idx

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

                # 从 batch outputs 中提取当前样本的 outputs
                sample_outputs_for_quad_conversion = {
                    'target_start_logits': outputs['target_start_logits'][i].unsqueeze(0),
                    'target_end_logits': outputs['target_end_logits'][i].unsqueeze(0),
                    'argument_start_logits': outputs['argument_start_logits'][i].unsqueeze(0),
                    'argument_end_logits': outputs['argument_end_logits'][i].unsqueeze(0),
                    'sequence_output': outputs['sequence_output'][i].unsqueeze(0),
                    'cls_output': outputs['cls_output'][i].unsqueeze(0),
                    'input_ids': input_ids[i].unsqueeze(0),  # 传递当前样本的 input_ids
                    'attention_mask': attention_mask[i].unsqueeze(0)  # 传递 attention_mask 用于 real_len
                }

                # print("t_s logits:", sample_outputs_for_quad_conversion['target_start_logits'].squeeze(1))
                # print("t_e logits:", sample_outputs_for_quad_conversion['target_end_logits'])
                # print("a_s logits:", sample_outputs_for_quad_conversion['argument_start_logits'])
                # print("a_e logits:", sample_outputs_for_quad_conversion['argument_end_logits'])

                sample_input_ids = input_ids[i].unsqueeze(0)  # [1, L]

                # 将当前样本的原始预测 logits 转换为四元组字符串列表
                predicted_quads_for_sample = convert_logits_to_quads(
                    sample_outputs_for_quad_conversion,
                    sample_input_ids,
                    tokenizer,
                    MAX_SEQ_LENGTH,
                    model  # 传入模型实例以便调用 Biaffine 和 classify_quad
                )
                if test_idx >= 80:
                    print("Predicted Quads for sample len:", len(predicted_quads_for_sample))

                all_predicted_quads.append(predicted_quads_for_sample)

                # 构造当前样本的“真实四元组字符串列表”
                true_quads_for_sample = []
                ids_i = input_ids[i]  # 当前样本的 token IDs
                current_attention_mask = attention_mask[i]  # 当前样本的 attention_mask
                real_len = int(current_attention_mask.sum().item())  # 真实 token 长度

                for quad in true_quads_batch[i]:
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

                if test_idx >= 80:
                    print("Predicted Quads for sample:", predicted_quads_for_sample)
                    print("True Quads for sample:", true_quads_for_sample)
                    test_idx = 0
                all_true_quads.append(true_quads_for_sample)

    metrics = calculate_f1_scores(all_predicted_quads, all_true_quads)
    return metrics


# -------------- convert_logits_to_quads --------------
def convert_logits_to_quads(outputs_for_a_sample: Dict[str, torch.Tensor],
                            sample_input_ids: torch.Tensor,  # 单样本的 input_ids
                            tokenizer: BertTokenizerFast,
                            max_seq_length: int,  # 实际上 seq_len 已经从 logits 中获取
                            model_instance: HateSpeechDetectionModel
                            ) -> List[str]:
    """
    将单个样本的 logits 转换为预测四元组（字符串形式）
    """
    global test_idx
    test_idx += 1
    predicted_quads_strings = []

    t_start_logits = outputs_for_a_sample['target_start_logits'].squeeze(0).squeeze(-1)  # [L]
    t_end_logits = outputs_for_a_sample['target_end_logits'].squeeze(0).squeeze(-1)
    a_start_logits = outputs_for_a_sample['argument_start_logits'].squeeze(0).squeeze(-1)
    a_end_logits = outputs_for_a_sample['argument_end_logits'].squeeze(0).squeeze(-1)
    sequence_output = outputs_for_a_sample['sequence_output']  # [1, L, H]
    cls_output = outputs_for_a_sample['cls_output']  # [1, H]
    attention_mask = outputs_for_a_sample['attention_mask'].squeeze(0)  # [L]

    seq_len = t_start_logits.size(0)

    # 修正：计算真实的文本长度 (排除 padding)
    real_len = int(attention_mask.sum().item())
    valid_span_start_idx = 1  # 跳过 [CLS]
    valid_span_end_idx = real_len - 2  # 跳过 [SEP] 和 padding

    if valid_span_end_idx < valid_span_start_idx:
        return []  # 文本太短，无法抽取 span (例如只有 [CLS] 和 [SEP])

    # 1. Span 识别：先计算概率
    t_start_probs = torch.sigmoid(t_start_logits)  # [L]
    t_end_probs = torch.sigmoid(t_end_logits)
    a_start_probs = torch.sigmoid(a_start_logits)
    a_end_probs = torch.sigmoid(a_end_logits)

    # 1b. Top-K Span Candidates
    k_span_actual = min(TOPK_SPAN, valid_span_end_idx - valid_span_start_idx + 1)
    if k_span_actual <= 0:
        return []

    topk_ts = torch.topk(t_start_probs[valid_span_start_idx: valid_span_end_idx + 1],
                         k=k_span_actual).indices + valid_span_start_idx
    topk_te = torch.topk(t_end_probs[valid_span_start_idx: valid_span_end_idx + 1],
                         k=k_span_actual).indices + valid_span_start_idx
    topk_as = torch.topk(a_start_probs[valid_span_start_idx: valid_span_end_idx + 1],
                         k=k_span_actual).indices + valid_span_start_idx
    topk_ae = torch.topk(a_end_probs[valid_span_start_idx: valid_span_end_idx + 1],
                         k=k_span_actual).indices + valid_span_start_idx

    candidate_target_spans = []
    candidate_argument_spans = []
    for ts in topk_ts.tolist():
        for te in topk_te.tolist():
            if ts <= te and (te - ts + 1) <= MAX_SPAN_LENGTH:
                score = t_start_probs[ts].item() * t_end_probs[te].item()  # 修正：使用乘积
                candidate_target_spans.append((ts, te, score))
    for as_idx in topk_as.tolist():
        for ae_idx in topk_ae.tolist():
            if as_idx <= ae_idx and (ae_idx - as_idx + 1) <= MAX_SPAN_LENGTH:
                score = a_start_probs[as_idx].item() * a_end_probs[ae_idx].item()  # 修正：使用乘积
                candidate_argument_spans.append((as_idx, ae_idx, score))

    candidate_target_spans.sort(key=lambda x: x[2], reverse=True)
    candidate_argument_spans.sort(key=lambda x: x[2], reverse=True)

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
    # 3. 筛选配对: 得分>阈值 或 Top-K

    N_t_cand, N_a_cand = pair_probs.size()

    candidate_pairs_scores = []
    for i_t in range(N_t_cand):
        for j_a in range(N_a_cand):
            score = pair_probs[i_t, j_a].item()
            candidate_pairs_scores.append((score, i_t, j_a))

    candidate_pairs_scores.sort(key=lambda x: x[0], reverse=True)

    # 设定一个预测四元组的最大数量，避免过多无效预测
    MAX_PREDICTED_QUADS_PER_SAMPLE = 5  # 可以根据需要调整

    for score, i_t, j_a in candidate_pairs_scores:
        ts, te = target_idx_map[i_t]
        as_idx, ae_idx = argument_idx_map[j_a]

        # Logic for skipping based on used_targets_tokens and used_arguments_tokens is removed.
        if score > PAIRING_THRESHOLD:  # 优先使用阈值过滤

            final_quad_candidates.append({
                't_start_token': ts,
                't_end_token': te,
                'a_start_token': as_idx,
                'a_end_token': ae_idx,
                'pair_score': score
            })
            # used_targets_tokens.add((ts, te)) # Removed
            # used_arguments_tokens.add((as_idx, ae_idx)) # Removed

            if len(final_quad_candidates) >= MAX_PREDICTED_QUADS_PER_SAMPLE:
                break  # 达到最大数量，停止添加

    if test_idx >= 80:
        print("candidates len:", len(final_quad_candidates))
    # 如果通过阈值后仍然没有候选，则强制取最高得分的 K_pair 个（防止完全没有预测）
    # Fallback logic no longer checks used_tokens
    if len(final_quad_candidates) == 0 and candidate_pairs_scores:
        K_pair_fallback = 1  # 至少预测一个, can be adjusted via Hype.py if needed
        for score, i_t, j_a in candidate_pairs_scores[:K_pair_fallback]:  # Iterate up to K_pair_fallback
            ts, te = target_idx_map[i_t]
            as_idx, ae_idx = argument_idx_map[j_a]

            # (ts, te) in used_targets_tokens or (as_idx, ae_idx) in used_arguments_tokens check removed

            final_quad_candidates.append({
                't_start_token': ts,
                't_end_token': te,
                'a_start_token': as_idx,
                'a_end_token': ae_idx,
                'pair_score': score
            })
            # used_targets_tokens.add((ts, te)) # Removed
            # used_arguments_tokens.add((as_idx, ae_idx)) # Removed
            # The MAX_PREDICTED_QUADS_PER_SAMPLE check will be applied after this loop

    # 再次排序并限制最终输出的四元组数量
    final_quad_candidates.sort(key=lambda x: x['pair_score'], reverse=True)
    final_quad_candidates = final_quad_candidates[:MAX_PREDICTED_QUADS_PER_SAMPLE]  # 再次裁剪

    # 4. 分类、收集信息用于NMS, (可选)应用NMS, 然后输出成字符串
    ids = sample_input_ids.squeeze(0)  # [L]

    if not final_quad_candidates:
        return []

    # NMS parameters are now imported from Hype.py via "from Hype import *"
    # APPLY_NMS, NMS_IOU_THRESHOLD_TARGET, NMS_IOU_THRESHOLD_ARGUMENT, NMS_CONTAINMENT_THRESHOLD
    # Sophisticated non-hate determination constants NON_HATE_DETERMINATION_MARGIN and 
    # MIN_SPECIFIC_HATE_GROUP_THRESHOLD are also imported from Hype.py via "from Hype import *"


    quads_for_nms = []

    # First, gather all necessary information for NMS
    if final_quad_candidates: # Ensure there are candidates before proceeding
        all_quad_reprs = []
        for quad_cand in final_quad_candidates:
            ts = quad_cand['t_start_token']
            te = quad_cand['t_end_token']
            as_idx = quad_cand['a_start_token']
            ae_idx = quad_cand['a_end_token']

            target_vec = model_instance._get_span_representation(sequence_output, ts, te, 0)
            argument_vec = model_instance._get_span_representation(sequence_output, as_idx, ae_idx, 0)
            combined_vec = torch.cat([target_vec, argument_vec, cls_output.squeeze(0)], dim=-1)
            all_quad_reprs.append(combined_vec)
        
        stacked_quad_reprs = torch.stack(all_quad_reprs, dim=0)

        group_logits_batch = model_instance.group_classifier(stacked_quad_reprs)
        hateful_logits_batch = model_instance.hateful_classifier(stacked_quad_reprs)
        group_probs_batch = torch.sigmoid(group_logits_batch)
        hateful_probs_batch = torch.sigmoid(hateful_logits_batch)

        for idx, quad_cand in enumerate(final_quad_candidates):
            ts = quad_cand['t_start_token']
            te = quad_cand['t_end_token']
            as_idx = quad_cand['a_start_token']
            ae_idx = quad_cand['a_end_token']
            score = quad_cand['pair_score']

            group_probs_single = group_probs_batch[idx] # Tensor of probabilities for each group
            hateful_prob_single = hateful_probs_batch[idx].item() # Scalar probability of "hate"

            # --- Step 1: Extract Necessary Probabilities ---
            prob_hateful_is_hate = hateful_prob_single
            prob_hateful_is_non_hate = 1.0 - hateful_prob_single

            non_hate_group_idx = -1
            others_group_idx = -1
            non_hate_group_label_str = "non-hate" # Fallback string
            others_group_label_str = "others"   # Fallback string

            try:
                non_hate_group_idx = TARGET_GROUP_CLASS_NAME.index(non_hate_group_label_str)
            except ValueError:
                # Log error or set flag: 'non-hate' not in TARGET_GROUP_CLASS_NAME
                # For now, we'll rely on non_hate_group_idx remaining -1 if not found.
                pass 

            try:
                others_group_idx = TARGET_GROUP_CLASS_NAME.index(others_group_label_str)
            except ValueError:
                # Log error or set flag: 'others' not in TARGET_GROUP_CLASS_NAME
                pass

            prob_group_is_non_hate = 0.0
            if non_hate_group_idx != -1:
                prob_group_is_non_hate = group_probs_single[non_hate_group_idx].item()

            max_specific_hate_prob = 0.0
            # Default strongest_specific_hate_label to 'others' string if index is valid, else the raw string.
            strongest_specific_hate_label = TARGET_GROUP_CLASS_NAME[others_group_idx] if others_group_idx != -1 else others_group_label_str
            
            num_specific_hate_candidate_groups = 0 # Not directly used in final decision logic for groups, but calculated as requested.

            for j in range(len(TARGET_GROUP_CLASS_NAME)):
                if j != non_hate_group_idx: # Specific hate group or 'others'
                    current_group_prob = group_probs_single[j].item()
                    if current_group_prob > max_specific_hate_prob:
                        max_specific_hate_prob = current_group_prob
                        strongest_specific_hate_label = TARGET_GROUP_CLASS_NAME[j]
                    if current_group_prob > MIN_SPECIFIC_HATE_GROUP_THRESHOLD: # Use imported constant
                        num_specific_hate_candidate_groups += 1
            
            # If no specific group had prob > 0, and 'others' is a valid group,
            # max_specific_hate_prob might effectively be the prob of 'others' if it was highest.
            # If others_group_idx is valid and strongest_specific_hate_label defaulted to it, ensure max_specific_hate_prob reflects its probability.
            if others_group_idx != -1 and strongest_specific_hate_label == TARGET_GROUP_CLASS_NAME[others_group_idx] and max_specific_hate_prob == 0.0:
                 max_specific_hate_prob = group_probs_single[others_group_idx].item()


            # --- Step 2: Implement New Decision Logic ---
            combined_score_non_hate = prob_hateful_is_non_hate + prob_group_is_non_hate
            combined_score_hate = prob_hateful_is_hate + max_specific_hate_prob
            
            final_hateful_str = ""
            final_predicted_groups_list = []

            if combined_score_non_hate >= combined_score_hate + NON_HATE_DETERMINATION_MARGIN: # Use imported constant
                final_hateful_str = "non-hate"
                final_predicted_groups_list = [TARGET_GROUP_CLASS_NAME[non_hate_group_idx] if non_hate_group_idx != -1 else non_hate_group_label_str]
            else: # Hate is dominant or margin not met
                final_hateful_str = "hate"
                temp_specific_groups = []
                for k in range(len(TARGET_GROUP_CLASS_NAME)):
                    if k != non_hate_group_idx and group_probs_single[k].item() > MIN_SPECIFIC_HATE_GROUP_THRESHOLD: # Use imported constant
                        temp_specific_groups.append(TARGET_GROUP_CLASS_NAME[k])
                
                if not temp_specific_groups:
                    final_predicted_groups_list = [TARGET_GROUP_CLASS_NAME[others_group_idx] if others_group_idx != -1 else others_group_label_str]
                else:
                    final_predicted_groups_list = temp_specific_groups

            # --- Step 3: Finalize group_str ---
            # Sort final_predicted_groups_list by the order in TARGET_GROUP_CLASS_NAME
            # Handle cases where a resolved group string might not be in TARGET_GROUP_CLASS_NAME (e.g. fallback "others")
            # For sorting, items not in TARGET_GROUP_CLASS_NAME can be placed at the end or beginning.
            # Here, we prioritize those in TARGET_GROUP_CLASS_NAME.
            final_predicted_groups_list.sort(key=lambda x: TARGET_GROUP_CLASS_NAME.index(x) if x in TARGET_GROUP_CLASS_NAME else float('inf'))
            final_group_str = ','.join(final_predicted_groups_list)

            # --- Step 4: Update `group_str` and `hateful_str` for NMS and output ---
            # These will be used in quads_for_nms.append below
            group_str = final_group_str 
            hateful_str = final_hateful_str

            target_text_raw = ""
            if ts != -1 and te != -1 and 0 <= ts <= te < seq_len:
                target_ids = ids[ts:te + 1].tolist()
                target_text_raw = tokenizer.decode(target_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                target_text_raw = target_text_raw.replace(" ", "") if target_text_raw.replace(" ", "") != "" else "NULL"
            else:
                target_text_raw = 'NULL'
            target_text_raw = target_text_raw if target_text_raw.strip() else 'NULL'
            
            argument_text_raw = ""
            if as_idx != -1 and ae_idx != -1 and 0 <= as_idx <= ae_idx < seq_len:
                argument_ids = ids[as_idx:ae_idx + 1].tolist()
                argument_text_raw = tokenizer.decode(argument_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                argument_text_raw = argument_text_raw.replace(" ", "") if argument_text_raw.replace(" ", "") != "" else "NULL"
            else:
                argument_text_raw = "NULL"
            argument_text_raw = argument_text_raw if argument_text_raw.strip() else 'NULL'

            quads_for_nms.append({
                't_start_token': ts, 't_end_token': te,
                'a_start_token': as_idx, 'a_end_token': ae_idx,
                'group_str': group_str,
                'hateful_str': hateful_str,
                'score': score,
                'target_text_raw': target_text_raw,
                'argument_text_raw': argument_text_raw
            })

    processed_quads = quads_for_nms
    if APPLY_NMS and quads_for_nms:
        processed_quads = apply_nms_to_quads(
            quads_for_nms,
            NMS_IOU_THRESHOLD_TARGET,
            NMS_IOU_THRESHOLD_ARGUMENT,
            NMS_CONTAINMENT_THRESHOLD # Use the imported constant
        )

    # Format final quads (after NMS) into strings
    for quad_data in processed_quads:
        predicted_quads_strings.append(
            f"{quad_data['target_text_raw']} | {quad_data['argument_text_raw']} | {quad_data['group_str']} | {quad_data['hateful_str']}"
        )

    if test_idx >= 80:
        print("Predicted Quads Strings len after NMS (if applied):", len(predicted_quads_strings))

    return predicted_quads_strings


# --- NMS Helper Function ---
def _calculate_span_iou(span1: Tuple[int, int], span2: Tuple[int, int]) -> float:
    """
    计算两个 Span (token indices) 之间的 IoU (Intersection over Union)。
    span 格式: (start_token_idx, end_token_idx)
    """
    s1, e1 = span1
    s2, e2 = span2

    # 计算交集
    intersection_start = max(s1, s2)
    intersection_end = min(e1, e2)

    intersection_length = max(0, intersection_end - intersection_start + 1)

    # 计算并集
    union_length = (e1 - s1 + 1) + (e2 - s2 + 1) - intersection_length

    if union_length == 0:
        return 0.0 # 避免除以零

    return intersection_length / union_length

# --- NMS Helper Functions (New/Modified) ---
def _spans_are_similar(span1: Tuple[int, int], span2: Tuple[int, int], iou_threshold: float, containment_threshold: float) -> bool:
    """
    Checks if two spans are similar based on IoU or containment.
    Handles invalid spans (start_token == -1) by returning False.
    """
    s1, e1 = span1
    s2, e2 = span2

    # If either span is invalid, they are not considered similar.
    if s1 == -1 or e1 == -1 or s2 == -1 or e2 == -1:
        return False
    
    # Ensure valid span ranges (s <= e)
    if s1 > e1 or s2 > e2:
        return False # Invalid span definition

    len1 = e1 - s1 + 1
    len2 = e2 - s2 + 1

    intersection_start = max(s1, s2)
    intersection_end = min(e1, e2)
    intersection_length = max(0, intersection_end - intersection_start + 1)

    if intersection_length == 0:
        return False

    union_length = len1 + len2 - intersection_length
    if union_length == 0: # Should not happen if intersection_length > 0, but as a safeguard
        return False 
        
    iou = intersection_length / union_length
    
    containment_score1 = intersection_length / len1 if len1 > 0 else 0
    containment_score2 = intersection_length / len2 if len2 > 0 else 0

    return (iou >= iou_threshold) or \
           (containment_score1 >= containment_threshold) or \
           (containment_score2 >= containment_threshold)


# --- NMS Functions ---
def _are_quads_similar(quad1: Dict, quad2: Dict, iou_threshold_target: float, iou_threshold_argument: float, containment_threshold: float) -> bool:
    """
    Checks if two quadruplets are similar based on group_str, hateful_str, 
    and combined IoU/containment of target/argument spans.
    """
    if quad1['group_str'] != quad2['group_str'] or \
       quad1['hateful_str'] != quad2['hateful_str']:
        return False

    # Target spans
    target_span1 = (quad1['t_start_token'], quad1['t_end_token'])
    target_span2 = (quad2['t_start_token'], quad2['t_end_token'])
    if not _spans_are_similar(target_span1, target_span2, iou_threshold_target, containment_threshold):
        return False

    # Argument spans
    arg_span1 = (quad1['a_start_token'], quad1['a_end_token'])
    arg_span2 = (quad2['a_start_token'], quad2['a_end_token'])
    if not _spans_are_similar(arg_span1, arg_span2, iou_threshold_argument, containment_threshold):
        return False
    
    return True


def apply_nms_to_quads(quad_list: List[Dict], iou_threshold_target: float, iou_threshold_argument: float, containment_threshold: float) -> List[Dict]:
    """
    Applies Non-Maximum Suppression (NMS) to a list of quadruplets.
    Each quad in quad_list is a dict with 't_start_token', 't_end_token',
    'a_start_token', 'a_end_token', 'group_str', 'hateful_str', and 'score'.
    """
    if not quad_list:
        return []

    # Sort quads by score in descending order
    # The 'score' key is assumed to be present in each quad dictionary
    quad_list.sort(key=lambda x: x.get('score', 0.0), reverse=True)

    kept_quads = []
    
    # Make a copy to allow modification during iteration
    remaining_quads = list(quad_list) 

    while remaining_quads:
        current_quad = remaining_quads.pop(0) # Get the quad with the highest score
        kept_quads.append(current_quad)

        # Filter out similar quads from the rest of the list
        temp_remaining_quads = []
        for quad_to_compare in remaining_quads:
            if not _are_quads_similar(current_quad, quad_to_compare, iou_threshold_target, iou_threshold_argument, containment_threshold):
                temp_remaining_quads.append(quad_to_compare)
        remaining_quads = temp_remaining_quads
        
    return kept_quads


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