# loss.py
import torch
import torch.nn as nn
from typing import Dict, List, Any
from iou_loss import SpanIoUWeightedLoss
from Hype import *  # 假设 Hype.py 中定义了这两个常量
import torch.nn.functional as F

# 损失函数实例
pos_weight_value = torch.tensor([(MAX_SEQ_LENGTH - 1) / 1], device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
span_loss_fct    = nn.BCEWithLogitsLoss(reduction='mean')
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

    # 2. 用 SpanIoUWeightedLoss 计算 Target 的 Span Loss
    span_iou_loss_fct = SpanIoUWeightedLoss(alpha=0.25, gamma=2.0, reduction="mean")
    target_span_loss  = span_iou_loss_fct(t_start_logits, t_end_logits, quads_labels_batch, device)

    # or IOU Span:
    argument_span_loss = span_iou_loss_fct(a_start_logits, a_end_logits, quads_labels_batch, device)

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
