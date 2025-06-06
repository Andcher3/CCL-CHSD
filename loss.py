# loss.py
import torch
import torch.nn as nn
from typing import Dict, List, Any
from iou_loss import SpanIoUWeightedLoss
from Hype import *  # 假设 Hype.py 中定义了这两个常量
import torch.nn.functional as F


def boundary_smoothing_span_loss(start_logits, end_logits, gold_spans_batch, seq_len, device):
    epsilon = BOUNDARY_SMOOTHING_EPSILON
    D_smooth = BOUNDARY_SMOOTHING_D
    # Not to be confused with MAX_SEQ_LENGTH, this is for span candidates.
    current_max_span_len = MAX_SPAN_LENGTH

    batch_size = start_logits.size(0)
    accumulated_loss = torch.tensor(0.0, device=device)
    num_gold_spans_processed = 0

    for b in range(batch_size):
        current_start_logits = start_logits[b]  # Shape: [seq_len]
        current_end_logits = end_logits[b]  # Shape: [seq_len]
        # gold_spans_batch[b] is a list of dicts, e.g., [{'start': s_val, 'end': e_val}, ...]
        current_gold_spans_info = gold_spans_batch[b]

        if not current_gold_spans_info:  # No gold spans for this batch item
            continue

        # 1. Generate all candidate spans and their raw scores for this batch item
        candidate_spans = []  # List of (s, e) tuples
        candidate_raw_scores = []  # List of corresponding scores
        for s_idx in range(seq_len):
            # Ensure e_idx does not exceed seq_len and respects current_max_span_len
            for e_idx in range(s_idx, min(seq_len, s_idx + current_max_span_len)):
                candidate_spans.append((s_idx, e_idx))
                score = current_start_logits[s_idx] + current_end_logits[e_idx]
                candidate_raw_scores.append(score)

        if not candidate_spans:  # No possible candidate spans generated
            continue

        candidate_scores_tensor = torch.tensor(candidate_raw_scores, device=device)
        # Predicted distribution (log probabilities) over all candidate spans for item b
        # Use .float() to ensure tensor is float for log_softmax if scores are int/long
        pred_log_probs = F.log_softmax(candidate_scores_tensor.float(), dim=0)

        # Iterate over each gold span for the current batch item
        for gold_span_info in current_gold_spans_info:
            gs = gold_span_info['start']
            ge = gold_span_info['end']

            # Ensure gs, ge are integers (they might be tensors from batch data)
            if isinstance(gs, torch.Tensor): gs = gs.item()
            if isinstance(ge, torch.Tensor): ge = ge.item()

            # Basic validation for gold span coordinates
            if not (0 <= gs < seq_len and 0 <= ge < seq_len and gs <= ge):
                continue

            # Check if the gold span is among the candidates
            try:
                gold_span_candidate_idx = candidate_spans.index((gs, ge))
            except ValueError:
                # This gold span is not a valid candidate under current MAX_SPAN_LENGTH
                # or other candidate generation rules. Skip it.
                continue

            # 2. Create smoothed target distribution for THIS gold span
            num_candidates = len(candidate_spans)
            target_dist = torch.zeros(num_candidates, device=device, dtype=torch.float)  # Ensure float for kl_div

            neighbors_indices = []
            for i, (cs, ce) in enumerate(candidate_spans):
                if i == gold_span_candidate_idx:
                    continue
                # Manhattan distance
                manhattan_dist = abs(gs - cs) + abs(ge - ce)
                if 0 < manhattan_dist <= D_smooth:  # Strictly positive distance
                    neighbors_indices.append(i)

            if not neighbors_indices:  # No neighbors within distance D_smooth
                target_dist[gold_span_candidate_idx] = 1.0
            else:
                target_dist[gold_span_candidate_idx] = 1.0 - epsilon
                if len(neighbors_indices) > 0:  # Distribute epsilon only if there are neighbors
                    prob_for_each_neighbor = epsilon / len(neighbors_indices)
                    for neighbor_idx in neighbors_indices:
                        target_dist[neighbor_idx] = prob_for_each_neighbor

            # Normalize target_dist to sum to 1, crucial for KL divergence
            current_sum = target_dist.sum()
            if current_sum > 1e-6:  # Avoid division by zero if target_dist is all zeros
                target_dist = target_dist / current_sum
            else:
                # This case handles when target_dist sums to 0 (e.g. epsilon=1, gold span is its only neighbor)
                # We should skip loss calculation for this gold span as target is undefined.
                continue

            # 3. Calculate KL Divergence loss for this gold span
            # Check for NaNs/Infs before calling kl_div
            if torch.isnan(target_dist).any() or torch.isinf(target_dist).any() or \
                    torch.isnan(pred_log_probs).any() or torch.isinf(pred_log_probs).any():
                # print(f"Warning: NaN or Inf detected before kl_div for gold span ({gs},{ge}). Skipping.")
                continue

            # Final check on target_dist sum before loss calculation
            if not (0.99 < target_dist.sum().item() < 1.01):
                # print(f"Warning: target_dist sum is not 1 before kl_div for gold span ({gs},{ge}). Sum: {
                # target_dist.sum().item()}. Skipping.")
                continue

            loss_one_gold_span = F.kl_div(pred_log_probs, target_dist, reduction='sum', log_target=False)

            if not torch.isnan(loss_one_gold_span) and not torch.isinf(loss_one_gold_span):
                accumulated_loss += loss_one_gold_span
                num_gold_spans_processed += 1
            # else:
            # print(f"Warning: NaN or Inf loss for gold span ({gs},{ge}). Skipping.")

    if num_gold_spans_processed == 0:
        return torch.tensor(0.0, device=device)

    return accumulated_loss / num_gold_spans_processed


# 损失函数实例 pos_weight_value = torch.tensor([(MAX_SEQ_LENGTH - 1) / 1], device=torch.device("cuda" if
# torch.cuda.is_available() else "cpu")) span_loss_fct    = nn.BCEWithLogitsLoss(reduction='mean')
span_loss_fct = nn.BCEWithLogitsLoss(reduction='none')  # reduction='none' 使得我们可以手动加权和求平均
group_loss_fct = nn.BCEWithLogitsLoss(reduction='mean')
hateful_loss_fct = nn.BCEWithLogitsLoss(reduction='mean')
pair_loss_fct = nn.BCEWithLogitsLoss(reduction='mean')


# ===== 3. 用到的辅助函数：为 Argument 定位标签 =====
def _make_arg_start_labels(quads_labels_batch, seq_len, device):
    B = len(quads_labels_batch)
    labels = torch.zeros((B, seq_len), device=device)
    for b, quads_list in enumerate(quads_labels_batch):
        for quad in quads_list:
            as_idx = quad['a_start_token'].item()
            if 0 <= as_idx < seq_len:
                labels[b, as_idx] = 1.0
    return labels


def _make_arg_end_labels(quads_labels_batch, seq_len, device):
    B = len(quads_labels_batch)
    labels = torch.zeros((B, seq_len), device=device)
    for b, quads_list in enumerate(quads_labels_batch):
        for quad in quads_list:
            ae_idx = quad['a_end_token'].item()
            if 0 <= ae_idx < seq_len:
                labels[b, ae_idx] = 1.0
    return labels


# ===== 4. compute_total_loss 的完整实现 =====
def compute_total_loss(outputs: Dict[str, torch.Tensor],
                       quads_labels_batch: List[List[Dict[str, Any]]],
                       model: nn.Module,
                       device: torch.device,
                       span_weight: float = 1.0,
                       group_weight: float = 0.5,
                       hateful_weight: float = 0.5,
                       biaffine_weight: float = 1.0
                       ) -> (torch.Tensor, Dict[str, float]):
    # 1. 从模型输出里拿到各 logits
    # shape: [B, L, 1] → squeeze → [B, L]
    t_start_logits = outputs['target_start_logits'].squeeze(-1)  # [B, L]
    t_end_logits = outputs['target_end_logits'].squeeze(-1)  # [B, L]
    a_start_logits = outputs['argument_start_logits'].squeeze(-1)  # [B, L]
    a_end_logits = outputs['argument_end_logits'].squeeze(-1)  # [B, L]
    sequence_output = outputs['sequence_output']  # [B, L, H]
    cls_output = outputs['cls_output']  # [B,   H]

    batch_size, seq_len = t_start_logits.size()

    # ---- Span Debug Logging for First Sample in Batch ----
    # if batch_size > 0:  # Ensure there's at least one sample
    #     print("\n---- Span Debug for Sample 0 ----")
    #     # Log True Spans for the first sample (b=0)
    #     true_target_spans_sample0 = []
    #     true_argument_spans_sample0 = []
    #     if quads_labels_batch and quads_labels_batch[0]:
    #         for quad in quads_labels_batch[0]:
    #             true_target_spans_sample0.append(
    #                 (quad['t_start_token'].item(), quad['t_end_token'].item())
    #             )
    #             true_argument_spans_sample0.append(
    #                 (quad['a_start_token'].item(), quad['a_end_token'].item())
    #             )
    #     print(f"True Target Spans (Sample 0): {true_target_spans_sample0}")
    #     print(f"True Argument Spans (Sample 0): {true_argument_spans_sample0}")
    #
    #     K_PRED_SPAN = 5
    #     safe_k_pred_span = min(K_PRED_SPAN, seq_len)  # Ensure k is not larger than seq_len
    #
    #     # Log Predicted Top-K Target Spans for the first sample (b=0)
    #     if safe_k_pred_span > 0:
    #         t_start_probs_sample0 = torch.sigmoid(t_start_logits[0])
    #         t_end_probs_sample0 = torch.sigmoid(t_end_logits[0])
    #
    #         topk_ts_indices_sample0 = torch.topk(t_start_probs_sample0, k=safe_k_pred_span).indices
    #         topk_te_indices_sample0 = torch.topk(t_end_probs_sample0, k=safe_k_pred_span).indices
    #
    #         pred_target_spans_cand_sample0 = []
    #         for ts in topk_ts_indices_sample0:
    #             for te in topk_te_indices_sample0:
    #                 if ts.item() <= te.item() and (te.item() - ts.item() + 1) <= MAX_SPAN_LENGTH:
    #                     score = t_start_probs_sample0[ts].item() * t_end_probs_sample0[te].item()
    #                     pred_target_spans_cand_sample0.append(((ts.item(), te.item()), score))
    #
    #         pred_target_spans_cand_sample0.sort(key=lambda x: x[1], reverse=True)
    #         print(
    #             f"Predicted Top-{K_PRED_SPAN} Target Spans (Sample 0): {pred_target_spans_cand_sample0[:K_PRED_SPAN]}")
    #
    #         # Log Predicted Top-K Argument Spans for the first sample (b=0)
    #         a_start_probs_sample0 = torch.sigmoid(a_start_logits[0])
    #         a_end_probs_sample0 = torch.sigmoid(a_end_logits[0])
    #
    #         topk_as_indices_sample0 = torch.topk(a_start_probs_sample0, k=safe_k_pred_span).indices
    #         topk_ae_indices_sample0 = torch.topk(a_end_probs_sample0, k=safe_k_pred_span).indices
    #
    #         pred_argument_spans_cand_sample0 = []
    #         for aus in topk_as_indices_sample0:  # Renamed to aus to avoid clash
    #             for aue in topk_ae_indices_sample0:  # Renamed to aue
    #                 if aus.item() <= aue.item() and (aue.item() - aus.item() + 1) <= MAX_SPAN_LENGTH:
    #                     score = a_start_probs_sample0[aus].item() * a_end_probs_sample0[aue].item()
    #                     pred_argument_spans_cand_sample0.append(((aus.item(), aue.item()), score))
    #
    #         pred_argument_spans_cand_sample0.sort(key=lambda x: x[1], reverse=True)
    #         print(
    #             f"Predicted Top-{K_PRED_SPAN} Argument Spans (Sample 0): {pred_argument_spans_cand_sample0[:K_PRED_SPAN]}")
    #     else:
    #         print(
    #             f"Predicted Top-{K_PRED_SPAN} Target Spans (Sample 0): Not enough tokens in sequence to predict {K_PRED_SPAN} spans.")
    #         print(
    #             f"Predicted Top-{K_PRED_SPAN} Argument Spans (Sample 0): Not enough tokens in sequence to predict {K_PRED_SPAN} spans.")
    #
    #     print("---- End of Span Debug for Sample 0 ----\n")
    # ---- End of Span Debug Logging ----

    # ---- 2. Span Extraction Loss using Boundary Smoothing ----
    # Prepare gold_spans_batch for Target spans
    gold_target_spans_batch = []
    for b_idx in range(batch_size):
        item_gold_target_spans = []
        if quads_labels_batch[b_idx]:  # Check if there are quads for this item
            for quad in quads_labels_batch[b_idx]:
                ts = quad['t_start_token']
                te = quad['t_end_token']
                if isinstance(ts, torch.Tensor): ts = ts.item()
                if isinstance(te, torch.Tensor): te = te.item()
                # Ensure spans are valid; -1 might indicate no span or padding
                if ts != -1 and te != -1 and ts <= te:
                    item_gold_target_spans.append({'start': ts, 'end': te})
        gold_target_spans_batch.append(item_gold_target_spans)

    # Prepare gold_spans_batch for Argument spans
    gold_argument_spans_batch = []
    for b_idx in range(batch_size):
        item_gold_argument_spans = []
        if quads_labels_batch[b_idx]:  # Check if there are quads for this item
            for quad in quads_labels_batch[b_idx]:
                als = quad['a_start_token']
                ale = quad['a_end_token']
                if isinstance(als, torch.Tensor): als = als.item()
                if isinstance(ale, torch.Tensor): ale = ale.item()
                # Ensure spans are valid
                if als != -1 and ale != -1 and als <= ale:
                    item_gold_argument_spans.append({'start': als, 'end': ale})
        gold_argument_spans_batch.append(item_gold_argument_spans)

    target_span_loss = boundary_smoothing_span_loss(
        t_start_logits, t_end_logits, gold_target_spans_batch, seq_len, device
    )
    argument_span_loss = boundary_smoothing_span_loss(
        a_start_logits, a_end_logits, gold_argument_spans_batch, seq_len, device
    )
    total_span_loss = target_span_loss + argument_span_loss

    # ---- 3. Biaffine 配对 Loss ----
    # 先在整个 batch 内解码候选 span
    all_target_spans = []  # 元素为 (batch_idx, ts, te)
    all_argument_spans = []  # 元素为 (batch_idx, as, ae)

    # 先把 k_span 限制到 [1, seq_len]
    k_span = min(TOPK_SPAN, seq_len)
    if k_span < 1:
        k_span = 1

    for b in range(batch_size):
        # 先计算每个位置的 start/end 概率
        ts_probs = torch.sigmoid(t_start_logits[b])  # [L]
        te_probs = torch.sigmoid(t_end_logits[b])
        as_probs = torch.sigmoid(a_start_logits[b])
        ae_probs = torch.sigmoid(a_end_logits[b])

        # Top-K，k_span <= seq_len
        topk_ts = torch.topk(ts_probs, k=k_span).indices.tolist()
        topk_te = torch.topk(te_probs, k=k_span).indices.tolist()
        topk_as = torch.topk(as_probs, k=k_span).indices.tolist()
        topk_ae = torch.topk(ae_probs, k=k_span).indices.tolist()

        # 枚举合法的 Target span
        for ts_idx in topk_ts:
            for te_idx in topk_te:
                if ts_idx <= te_idx and (te_idx - ts_idx + 1) <= MAX_SPAN_LENGTH:
                    all_target_spans.append((b, ts_idx, te_idx))
        # 枚举合法的 Argument span
        for as_idx in topk_as:
            for ae_idx in topk_ae:
                if as_idx <= ae_idx and (ae_idx - as_idx + 1) <= MAX_SPAN_LENGTH:
                    all_argument_spans.append((b, as_idx, ae_idx))

    # 从 sequence_output 中提取所有候选 span 的向量
    N_t = len(all_target_spans)
    N_a = len(all_argument_spans)
    # FOR TEST
    # print(f"N_t (number of candidate target spans): {N_t}")
    # print(f"N_a (number of candidate argument spans): {N_a}")
    # FOR TEST
    if N_t > 0 and N_a > 0:
        tgt_vecs = []
        for (b, ts, te) in all_target_spans:
            vec = model._get_span_representation(sequence_output, ts, te, b)  # [H]
            tgt_vecs.append(vec)
        arg_vecs = []
        for (b, as_, ae) in all_argument_spans:
            vec = model._get_span_representation(sequence_output, as_, ae, b)
            arg_vecs.append(vec)

        target_mat = torch.stack(tgt_vecs, dim=0)  # [N_t, H]
        argument_mat = torch.stack(arg_vecs, dim=0)  # [N_a, H]

        # Biaffine 打分
        pair_logits = model.biaffine_pairing(target_mat, argument_mat)  # [N_t, N_a]
        pair_probs = torch.sigmoid(pair_logits)

        # 构建 Gold Pair 标签
        gold_pair_labels = torch.zeros_like(pair_logits, device=device)
        for i_idx, (b_t, ts, te) in enumerate(all_target_spans):
            for j_idx, (b_a, as_, ae) in enumerate(all_argument_spans):
                if b_t == b_a:
                    for quad in quads_labels_batch[b_t]:
                        if (quad['t_start_token'] == ts and quad['t_end_token'] == te and
                                quad['a_start_token'] == as_ and quad['a_end_token'] == ae):
                            gold_pair_labels[i_idx, j_idx] = 1.0
                            break

        # print(f"Number of positive gold pairs: {torch.sum(gold_pair_labels).item()}")

        # =================FOR TEST===========================================
        # true_quads_found_in_candidates = 0
        # total_true_quads = 0
        # for b, quads_list in enumerate(quads_labels_batch):
        #     total_true_quads += len(quads_list)
        #     for quad in quads_list:
        #         true_ts, true_te = quad['t_start_token'].item(), quad['t_end_token'].item()
        #         true_as, true_ae = quad['a_start_token'].item(), quad['a_end_token'].item()
        #
        #         is_target_found = any(
        #             cand_b == b and cand_ts == true_ts and cand_te == true_te for cand_b, cand_ts, cand_te in
        #             all_target_spans)
        #         is_argument_found = any(
        #             cand_b == b and cand_as == true_as and cand_ae == true_ae for cand_b, cand_as, cand_ae in
        #             all_argument_spans)
        #
        #         if is_target_found and is_argument_found:
        #             true_quads_found_in_candidates += 1
        # print(f"True quads found in candidates: {true_quads_found_in_candidates}")
        # print(f"Total true quads in batch: {total_true_quads}")
        # =================FOR TEST===========================================

        # 计算 BCEWithLogitsLoss
        total_pair_loss = pair_loss_fct(pair_logits, gold_pair_labels)
    else:
        total_pair_loss = torch.tensor(0.0, device=device)

    # ---- 4. Classification Loss (Group + Hate) ----
    # 一次性把所有 gold 四元组做批量化
    all_span_reprs = []
    all_group_labels = []
    all_hateful_labels = []
    for b, quads_list in enumerate(quads_labels_batch):
        for quad in quads_list:
            ts, te = quad['t_start_token'], quad['t_end_token']
            as_, ae = quad['a_start_token'], quad['a_end_token']
            tgt_vec = model._get_span_representation(sequence_output, ts, te, b)  # [H]
            arg_vec = model._get_span_representation(sequence_output, as_, ae, b)  # [H]
            cls_vec = cls_output[b]  # [H]
            all_span_reprs.append(torch.cat([tgt_vec, arg_vec, cls_vec], dim=-1))  # [3H]
            all_group_labels.append(torch.tensor(quad['group_vector'], device=device, dtype=torch.float))
            all_hateful_labels.append(torch.tensor([quad['hateful_flag']], device=device, dtype=torch.float))

    if all_span_reprs:
        span_cat = torch.stack(all_span_reprs, dim=0)  # [N_q, 3H]
        group_logits = model.group_classifier(span_cat)  # [N_q, num_groups]
        hateful_logits = model.hateful_classifier(span_cat)  # [N_q, 1]

        all_group_labels = torch.stack(all_group_labels, dim=0)  # [N_q, num_groups]
        all_hateful_labels = torch.stack(all_hateful_labels, dim=0).view(-1)  # [N_q]

        total_group_loss = group_loss_fct(group_logits, all_group_labels)
        total_hateful_loss = hateful_loss_fct(hateful_logits.view(-1), all_hateful_labels)
    else:
        total_group_loss = torch.tensor(0.0, device=device)
        total_hateful_loss = torch.tensor(0.0, device=device)

    # ---- 5. 加权合并 ----
    loss = (
            span_weight * total_span_loss +
            biaffine_weight * total_pair_loss +
            group_weight * total_group_loss +
            hateful_weight * total_hateful_loss
    )

    # 准备返回的各组件 Loss（转为 Python float 方便打印）
    loss_components = {
        'span_loss': total_span_loss.item(),
        'biaffine_loss': total_pair_loss.item(),
        'group_loss': total_group_loss.item(),
        'hateful_loss': total_hateful_loss.item()
    }
    return loss, loss_components
