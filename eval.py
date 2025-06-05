# eval.py

import torch
import torch.nn.functional as F
import difflib
from typing import List, Dict, Any
from transformers import BertTokenizerFast
import numpy as np
from torch.utils.data import DataLoader

from Model import HateSpeechDetectionModel
from Hype import *


# ----------------- evaluate_model -----------------
def evaluate_model(model: HateSpeechDetectionModel,
                   val_dataloader: DataLoader,
                   tokenizer: BertTokenizerFast,
                   device: torch.device) -> Dict[str, float]:
    """
    在验证集上评估模型性能，计算硬匹配和软匹配 F1 分数。
    """
    model.eval()
    all_predicted_quads = []
    all_true_quads = []

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)  # [B, L]
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            true_quads_batch = batch['quads_labels']  # List[List[quad_dict]]，batch 内保持在 CPU

            # 原始 logits + hidden outputs
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
            # print("raw output:", outputs)
            batch_size, seq_len = input_ids.size()

            for i in range(batch_size):
                # 单样本 logits (& hidden) 封装
                sample_outputs = {
                    'target_start_logits': outputs['target_start_logits'][i].unsqueeze(0),
                    'target_end_logits': outputs['target_end_logits'][i].unsqueeze(0),
                    'argument_start_logits': outputs['argument_start_logits'][i].unsqueeze(0),
                    'argument_end_logits': outputs['argument_end_logits'][i].unsqueeze(0),
                    'sequence_output': outputs['sequence_output'][i].unsqueeze(0),
                    'cls_output': outputs['cls_output'][i].unsqueeze(0),
                }

                # **注意：这里必须把 sample_input_ids 也传进去**
                sample_input_ids = input_ids[i].unsqueeze(0)  # [1, L]

                predicted_quads_for_sample = convert_logits_to_quads(
                    sample_outputs,
                    sample_input_ids,
                    tokenizer,
                    MAX_SEQ_LENGTH,
                    model
                )
                all_predicted_quads.append(predicted_quads_for_sample)

                # 构造当前样本的“真实四元组字符串列表”
                true_quads_for_sample = []
                ids_i = input_ids[i]
                for quad in true_quads_batch[i]:
                    ts, te = quad['t_start_token'].item(), quad['t_end_token'].item()
                    as_, ae = quad['a_start_token'].item(), quad['a_end_token'].item()
                    target_text = ""
                    argument_text = ""
                    if 0 <= ts <= te < seq_len:
                        raw = tokenizer.decode(
                            ids_i[ts:te + 1], skip_special_tokens=True,
                            clean_up_tokenization_spaces=False
                        )
                        target_text = raw.replace(" ", "") if raw.replace(" ", "") != "" else "NULL"
                    else:
                        target_text = "NULL"

                    if 0 <= as_ <= ae < seq_len:
                        raw = tokenizer.decode(
                            ids_i[as_:ae + 1], skip_special_tokens=True,
                            clean_up_tokenization_spaces=False
                        )
                        argument_text = raw.replace(" ", "") if raw.replace(" ", "") != "" else "NULL"
                    else:
                        argument_text = "NULL"

                    group_labels = [TARGET_GROUP_CLASS_NAME[idx]
                                    for idx, val in enumerate(quad['group_vector'].tolist()) if val == 1]
                    group_str = ",".join(group_labels) if group_labels else "non-hate"
                    hateful_str = "hate" if quad['hateful_flag'].item() == 1 else "non-hate"

                    true_quads_for_sample.append(
                        f"{target_text} | {argument_text} | {group_str} | {hateful_str}"
                    )

                print("pred vs true:\n")
                print(predicted_quads_for_sample)
                print(true_quads_for_sample)
                all_true_quads.append(true_quads_for_sample)

    metrics = calculate_f1_scores(all_predicted_quads, all_true_quads)
    return metrics


# -------------- convert_logits_to_quads --------------
def convert_logits_to_quads(outputs_for_a_sample: Dict[str, torch.Tensor],
                            sample_input_ids: torch.Tensor,
                            tokenizer: BertTokenizerFast,
                            max_seq_length: int,
                            model_instance: HateSpeechDetectionModel
                            ) -> List[str]:
    """
    将单个样本的 logits 转换为预测四元组（字符串形式）
    """
    predicted_quads_strings = []

    # 从 outputs_for_a_sample 中提取
    t_start_logits = outputs_for_a_sample['target_start_logits'].squeeze(0).squeeze(-1)
    t_end_logits = outputs_for_a_sample['target_end_logits'].squeeze(0).squeeze(-1)
    a_start_logits = outputs_for_a_sample['argument_start_logits'].squeeze(0).squeeze(-1)
    a_end_logits = outputs_for_a_sample['argument_end_logits'].squeeze(0).squeeze(-1)
    sequence_output = outputs_for_a_sample['sequence_output']  # [1, L, H]
    cls_output = outputs_for_a_sample['cls_output']  # [1, H]

    seq_len = t_start_logits.size(0)

    # 1. Span 识别：先计算概率
    t_start_probs = torch.sigmoid(t_start_logits)  # [L]
    t_end_probs = torch.sigmoid(t_end_logits)
    a_start_probs = torch.sigmoid(a_start_logits)
    a_end_probs = torch.sigmoid(a_end_logits)
    # print('t_start_probs\n', t_start_probs)

    # 1a. 计算文本实际 token 范围
    # 通过 attention_mask 找到真实长度
    # 这一步假设 sample_input_ids 中的 padding token 已标注为 0 或 [PAD]
    # 但注意：outputs_for_a_sample 并未包含 attention_mask，所以这里我们用一个小 trick：
    # 我们相信“只要 logits 在非有效 token 那里接近 -inf 或 0，就不会被 topk 选中”。
    # 针对安全起见，可以假定 seq_len 就是真实长度，或者提前把有效长度传进来。
    valid_span_start_idx = 1  # 跳过 [CLS]
    valid_span_end_idx = seq_len - 2  # 跳过最后的 [SEP]

    if valid_span_end_idx < valid_span_start_idx:
        return []  # 文本太短，无法抽取 span

    # 1b. Top-K Pap
    k_span = min(TOPK_SPAN, valid_span_end_idx - valid_span_start_idx + 1)
    k_span = max(k_span, 1)

    # 选 Top-K 的 start/end 索引
    topk_ts = torch.topk(t_start_probs[valid_span_start_idx: valid_span_end_idx + 1],
                         k=k_span).indices + valid_span_start_idx
    topk_te = torch.topk(t_end_probs[valid_span_start_idx: valid_span_end_idx + 1],
                         k=k_span).indices + valid_span_start_idx
    topk_as = torch.topk(a_start_probs[valid_span_start_idx: valid_span_end_idx + 1],
                         k=k_span).indices + valid_span_start_idx
    topk_ae = torch.topk(a_end_probs[valid_span_start_idx: valid_span_end_idx + 1],
                         k=k_span).indices + valid_span_start_idx
    # print('topk_ts\n', topk_ts)
    # print('.tolist()\n', topk_ts.tolist())
    # 枚举合法 Span
    candidate_target_spans = []
    candidate_argument_spans = []
    for ts in topk_ts.tolist():
        for te in topk_te.tolist():
            if ts <= te and (te - ts + 1) <= MAX_SPAN_LENGTH:
                score = t_start_probs[ts].item() + t_end_probs[te].item()
                candidate_target_spans.append((ts, te, score))
    for as_idx in topk_as.tolist():
        for ae_idx in topk_ae.tolist():
            if as_idx <= ae_idx and (ae_idx - as_idx + 1) <= MAX_SPAN_LENGTH:
                score = a_start_probs[as_idx].item() + a_end_probs[ae_idx].item()
                candidate_argument_spans.append((as_idx, ae_idx, score))

    if not candidate_target_spans or not candidate_argument_spans:
        print('candidate_target_spans, candidate_argument_spans', candidate_target_spans, candidate_argument_spans)
        return []

    # 按 Score 排序并选前 few
    candidate_target_spans.sort(key=lambda x: x[2], reverse=True)
    candidate_argument_spans.sort(key=lambda x: x[2], reverse=True)
    # 你可以把最终候选再裁剪成 Top_K_FINAL 个，看需求
    # 例如：
    # candidate_target_spans   = candidate_target_spans[:TOPK_FINAL]
    # candidate_argument_spans = candidate_argument_spans[:TOPK_FINAL]

    # 2. Biaffine 配对
    target_vecs = []
    target_idx_map = []
    for (ts, te, _) in candidate_target_spans:
        vec = model_instance._get_span_representation(sequence_output, ts, te, batch_idx=0)
        target_vecs.append(vec)
        target_idx_map.append((ts, te))

    argument_vecs = []
    argument_idx_map = []
    for (as_idx, ae_idx, _) in candidate_argument_spans:
        vec = model_instance._get_span_representation(sequence_output, as_idx, ae_idx, batch_idx=0)
        argument_vecs.append(vec)
        argument_idx_map.append((as_idx, ae_idx))

    if not target_vecs or not argument_vecs:
        print('target_vecs, argument_vecs', target_vecs, argument_vecs)
        return []

    target_mat = torch.stack(target_vecs, dim=0)  # [N_t, H]
    argument_mat = torch.stack(argument_vecs, dim=0)  # [N_a, H]

    pair_logits = model_instance.biaffine_pairing(target_mat, argument_mat)  # [N_t, N_a]
    pair_probs = torch.sigmoid(pair_logits)

    # 3. 筛选配对: 得分>阈值 或 Top-K
    pairing_threshold = 0.5
    final_quad_candidates = []
    N_t, N_a = pair_probs.size()
    K_pair = 1
    # print('N_t， N_a', N_t, N_a)
    for i_t in range(N_t):
        for j_a in range(N_a):
            if pair_probs[i_t, j_a].item() > pairing_threshold:
                ts, te = target_idx_map[i_t]
                as_idx, ae_idx = argument_idx_map[j_a]
                final_quad_candidates.append({
                    't_start_token': ts,
                    't_end_token': te,
                    'a_start_token': as_idx,
                    'a_end_token': ae_idx,
                    'pair_score': pair_probs[i_t, j_a].item()
                })

    # 如果筛出来还是空，则退到 Top-K 策略
    if len(final_quad_candidates) == 0:
        pair_scores_flat = pair_probs.view(-1)
        k_eff = min(K_pair, pair_scores_flat.size(0))
        topk_flat_idxs = torch.topk(pair_scores_flat, k=k_eff).indices.tolist()
        for flat_idx in topk_flat_idxs:
            i_t = flat_idx // N_a
            j_a = flat_idx % N_a
            ts, te = target_idx_map[i_t]
            as_idx, ae_idx = argument_idx_map[j_a]
            score = pair_probs[i_t, j_a].item()
            final_quad_candidates.append({
                't_start_token': ts,
                't_end_token': te,
                'a_start_token': as_idx,
                'a_end_token': ae_idx,
                'pair_score': score
            })

    # 按 pair_score 排序、可以再 crop Top-K
    # print('final_quad_candidates', final_quad_candidates)
    final_quad_candidates.sort(key=lambda x: x['pair_score'], reverse=True)

    # 4. 分类并输出成字符串
    ids = sample_input_ids.squeeze(0)  # [L]
    for quad_cand in final_quad_candidates:
        ts, te = quad_cand['t_start_token'], quad_cand['t_end_token']
        as_idx, ae_idx = quad_cand['a_start_token'], quad_cand['a_end_token']

        # 分别调用 classify_quad
        group_logits_single, hateful_logits_single = model_instance.classify_quad(
            sequence_output=sequence_output,
            cls_output=cls_output,
            t_start_token=ts,
            t_end_token=te,
            a_start_token=as_idx,
            a_end_token=ae_idx,
            batch_idx=0
        )

        group_probs = torch.sigmoid(group_logits_single).squeeze(0)  # [num_groups]
        hateful_prob = torch.sigmoid(hateful_logits_single).item()  # float

        predicted_groups = []
        for idx, prob in enumerate(group_probs.tolist()):
            if prob > 0.5:
                predicted_groups.append(TARGET_GROUP_CLASS_NAME[idx])
        group_str = ",".join(predicted_groups) if predicted_groups else "non-hate"
        hateful_str = "hate" if hateful_prob > 0.5 else "non-hate"

        # 把 token 片段 decode 成文本
        target_text = ""
        argument_text = ""
        if 0 <= ts <= te < seq_len:
            raw = tokenizer.decode(
                ids[ts:te + 1], skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            target_text = raw.replace(" ", "") if raw.replace(" ", "") != "" else "NULL"

        if 0 <= as_idx <= ae_idx < seq_len:
            raw = tokenizer.decode(
                ids[as_idx:ae_idx + 1], skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            argument_text = raw.replace(" ", "") if raw.replace(" ", "") != "" else "NULL"

        # 仍然保留四元组之间的 “ | ” 空格分隔
        predicted_quads_strings.append(
            f"{target_text} | {argument_text} | {group_str} | {hateful_str}"
        )

    return predicted_quads_strings


# --------------- calculate_f1_scores ---------------
def calculate_f1_scores(pred_quads_list: List[List[str]], true_quads_list: List[List[str]]) -> Dict[str, float]:
    """
    计算硬匹配和软匹配 F1 分数（假设每个样本的 true_quads_list[i] 至少有一个元素）。
    """
    tp_hard, fp_hard, fn_hard = 0, 0, 0
    tp_soft, fp_soft, fn_soft = 0, 0, 0

    assert len(pred_quads_list) == len(true_quads_list)

    for preds_for_sample, truths_for_sample in zip(pred_quads_list, true_quads_list):
        # 假设 truths_for_sample 至少包含一条四元组
        # --- 硬匹配 ---
        matched_preds_hard  = [False] * len(preds_for_sample)
        matched_truths_hard = [False] * len(truths_for_sample)

        for p_idx, p_str in enumerate(preds_for_sample):
            for t_idx, t_str in enumerate(truths_for_sample):
                if p_str == t_str and not matched_truths_hard[t_idx]:
                    tp_hard += 1
                    matched_preds_hard[p_idx]  = True
                    matched_truths_hard[t_idx] = True
                    break

        fp_hard += sum(1 for matched in matched_preds_hard if not matched)
        fn_hard += sum(1 for matched in matched_truths_hard if not matched)

        # --- 软匹配 ---
        parsed_preds  = [_parse_quad_string(x) for x in preds_for_sample]
        parsed_truths = [_parse_quad_string(x) for x in truths_for_sample]
        matched_preds_soft  = [False] * len(parsed_preds)
        matched_truths_soft = [False] * len(parsed_truths)

        for p_idx, pred_quad in enumerate(parsed_preds):
            for t_idx, true_quad in enumerate(parsed_truths):
                if not matched_truths_soft[t_idx] and _is_soft_match(pred_quad, true_quad):
                    tp_soft += 1
                    matched_preds_soft[p_idx]  = True
                    matched_truths_soft[t_idx] = True
                    break

        fp_soft += sum(1 for m in matched_preds_soft  if not m)
        fn_soft += sum(1 for m in matched_truths_soft if not m)

    # 计算硬指标
    hard_precision = tp_hard / (tp_hard + fp_hard) if (tp_hard + fp_hard) > 0 else 0.0
    hard_recall    = tp_hard / (tp_hard + fn_hard) if (tp_hard + fn_hard) > 0 else 0.0
    hard_f1        = (2 * hard_precision * hard_recall) / (hard_precision + hard_recall) \
                        if (hard_precision + hard_recall) > 0 else 0.0

    # 计算软指标
    soft_precision = tp_soft / (tp_soft + fp_soft) if (tp_soft + fp_soft) > 0 else 0.0
    soft_recall    = tp_soft / (tp_soft + fn_soft) if (tp_soft + fn_soft) > 0 else 0.0
    soft_f1        = (2 * soft_precision * soft_recall) / (soft_precision + soft_recall) \
                        if (soft_precision + soft_recall) > 0 else 0.0

    avg_f1 = (hard_f1 + soft_f1) / 2.0

    return {
        "hard_precision": hard_precision,
        "hard_recall":    hard_recall,
        "hard_f1":        hard_f1,
        "soft_precision": soft_precision,
        "soft_recall":    soft_recall,
        "soft_f1":        soft_f1,
        "average_f1":     avg_f1
    }


# --- 辅助函数 ---
def _parse_quad_string(quad_str: str) -> Dict[str, str]:
    parts = quad_str.split(" | ")
    if len(parts) == 4:
        return {
            'target': parts[0],
            'argument': parts[1],
            'group': parts[2],
            'hateful': parts[3]
        }
    return {'target': '', 'argument': '', 'group': '', 'hateful': ''}


# 超过50%软匹配
def _is_soft_match(pred_quad: Dict[str, str], true_quad: Dict[str, str], similarity_threshold: float = 0.5) -> bool:
    if pred_quad['group'] != true_quad['group'] or pred_quad['hateful'] != true_quad['hateful']:
        return False
    t_sim = difflib.SequenceMatcher(None, pred_quad['target'], true_quad['target']).ratio()
    a_sim = difflib.SequenceMatcher(None, pred_quad['argument'], true_quad['argument']).ratio()
    return (t_sim >= similarity_threshold) and (a_sim >= similarity_threshold)


# --- 快速测试 calculate_f1_scores ---
if __name__ == '__main__':
    print("--- Testing calculate_f1_scores ---")
    pred1 = ["t1 | a1 | Racism, Sexism | hate"]
    true1 = ["t1 | a1 | Racism | hate"]
    pred2 = ["A | B | Sexism | non-hate"]
    true2 = ["A | B | Sexism | non-hate", "C | D_wrong | Sexism | non-hate"]
    pred3 = ["CCC | DDD | LGBTQ | hate"]
    true3 = ["C | D | LGBTQ | hate"]
    pred4 = ["X | Y | Racism | non-hate"]
    true4 = ["X | Y | Racism | non-hate"]
    pred5 = ["Z | W | Sexism | hate"]
    true5 = ["Z | W | Sexism | hate"]

    all_predicted = [pred1, pred2, pred3, pred4, pred5]
    all_true = [true1, true2, true3, true4, true5]
    metrics = calculate_f1_scores(all_predicted, all_true)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
