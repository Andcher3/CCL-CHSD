# loss.py
import torch
import torch.nn as nn
from typing import Dict, List, Any
from iou_loss import SpanIoUWeightedLoss
from Hype import *  # 假设 Hype.py 中定义了这两个常量
import torch.nn.functional as F

# 损失函数实例
# pos_weight_value = torch.tensor([(MAX_SEQ_LENGTH - 1) / 1], device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
# span_loss_fct    = nn.BCEWithLogitsLoss(reduction='mean')
span_loss_fct = nn.BCEWithLogitsLoss(reduction='none') # reduction='none' 使得我们可以手动加权和求平均
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
                       span_weight: float    = 1.0,
                       group_weight: float   = 0.5,
                       hateful_weight: float = 0.5,
                       biaffine_weight: float= 1.0
                       ) -> (torch.Tensor, Dict[str, float]):

    # 1. 从模型输出里拿到各 logits
    # shape: [B, L, 1] → squeeze → [B, L]
    t_start_logits = outputs['target_start_logits'].squeeze(-1)   # [B, L]
    t_end_logits   = outputs['target_end_logits'].squeeze(-1)     # [B, L]
    a_start_logits = outputs['argument_start_logits'].squeeze(-1) # [B, L]
    a_end_logits   = outputs['argument_end_logits'].squeeze(-1)   # [B, L]
    sequence_output = outputs['sequence_output']  # [B, L, H]
    cls_output      = outputs['cls_output']       # [B,   H]

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
    def iou_span_loss():
        # 2. 用 SpanIoUWeightedLoss 计算 Target 的 Span Loss
        target_span_loss_fct = SpanIoUWeightedLoss(alpha=0.1, gamma=2.0, reduction="mean", span_type="target")
        argument_span_loss_fct = SpanIoUWeightedLoss(alpha=0.1, gamma=2.0, reduction="mean", span_type="argument")

        target_span_loss = target_span_loss_fct(t_start_logits, t_end_logits, quads_labels_batch, device)
        argument_span_loss = argument_span_loss_fct(a_start_logits, a_end_logits, quads_labels_batch, device)

        total_span_loss = target_span_loss + argument_span_loss
    def bce_span_loss():
        # ---- 2. Span 抽取 Loss (使用 BCEWithLogitsLoss + 动态 pos_weight) ----
        # 构造 token-level 真实标签
        t_start_labels = torch.zeros((batch_size, seq_len), device=device)
        t_end_labels = torch.zeros((batch_size, seq_len), device=device)
        a_start_labels = torch.zeros((batch_size, seq_len), device=device)
        a_end_labels = torch.zeros((batch_size, seq_len), device=device)

        for b, quads_list in enumerate(quads_labels_batch):
            for quad in quads_list:
                ts = quad['t_start_token'].item()
                te = quad['t_end_token'].item()
                as_ = quad['a_start_token'].item()
                ae = quad['a_end_token'].item()

                # 为所有有效的 Span 起点/终点设置标签 1.0
                if 0 <= ts < seq_len:
                    t_start_labels[b, ts] = 1.0
                if 0 <= te < seq_len:
                    t_end_labels[b, te] = 1.0
                if 0 <= as_ < seq_len:
                    a_start_labels[b, as_] = 1.0
                if 0 <= ae < seq_len:
                    a_end_labels[b, ae] = 1.0

        # 动态计算 pos_weight (负样本数 / 正样本数)
        # 针对 Target Start
        pos_count_ts = t_start_labels.sum(dim=1)  # [B]
        neg_count_ts = seq_len - pos_count_ts
        pos_weight_ts = torch.where(pos_count_ts > 0, neg_count_ts / (pos_count_ts + 1e-5),
                                    torch.tensor(1.0, device=device))
        pos_weight_ts = pos_weight_ts.unsqueeze(-1).expand_as(t_start_logits)  # [B, L]

        # 针对 Target End
        pos_count_te = t_end_labels.sum(dim=1)
        neg_count_te = seq_len - pos_count_te
        pos_weight_te = torch.where(pos_count_te > 0, neg_count_te / (pos_count_te + 1e-5),
                                    torch.tensor(1.0, device=device))
        pos_weight_te = pos_weight_te.unsqueeze(-1).expand_as(t_end_logits)

        # 针对 Argument Start
        pos_count_as = a_start_labels.sum(dim=1)
        neg_count_as = seq_len - pos_count_as
        pos_weight_as = torch.where(pos_count_as > 0, neg_count_as / (pos_count_as + 1e-5),
                                    torch.tensor(1.0, device=device))
        pos_weight_as = pos_weight_as.unsqueeze(-1).expand_as(a_start_logits)

        # 针对 Argument End
        pos_count_ae = a_end_labels.sum(dim=1)
        neg_count_ae = seq_len - pos_count_ae
        pos_weight_ae = torch.where(pos_count_ae > 0, neg_count_ae / (pos_count_ae + 1e-5),
                                    torch.tensor(1.0, device=device))
        pos_weight_ae = pos_weight_ae.unsqueeze(-1).expand_as(a_end_logits)
        span_loss_ts = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight_ts)
        span_loss_te = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight_te)
        span_loss_as = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight_as)
        span_loss_ae = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight_ae)

        # 计算损失并求平均 (reduction='none' 后手动求 mean)
        total_span_loss = (
                (span_loss_ts(t_start_logits, t_start_labels)).mean() +
                (span_loss_te(t_end_logits, t_end_labels)).mean() +
                (span_loss_as(a_start_logits, a_start_labels)).mean() +
                (span_loss_ae(a_end_logits, a_end_labels)).mean()
        )
        return total_span_loss
    total_span_loss = bce_span_loss()

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
