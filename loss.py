# loss.py
import torch
import torch.nn as nn
from typing import Dict, List, Any, Tuple
from iou_loss import SpanIoUWeightedLoss, SpanBoundarySmoothKLDivLoss
from Hype import *  # 假设 Hype.py 中定义了这两个常量
import torch.nn.functional as F


target_span_iou_loss_fct = SpanIoUWeightedLoss(
    alpha=BOUNDARY_SMOOTHING_EPSILON,  # 对应 IoU Loss 的 alpha
    gamma=BOUNDARY_SMOOTHING_D,  # 对应 IoU Loss 的 gamma
    reduction="mean",
    span_type="target"
)
argument_span_iou_loss_fct = SpanIoUWeightedLoss(
    alpha=BOUNDARY_SMOOTHING_EPSILON,
    gamma=BOUNDARY_SMOOTHING_D,
    reduction="mean",
    span_type="argument"
)

# 修正：实例化 SpanBoundarySmoothKLDivLoss 用于 Target 和 Argument Span
target_span_kl_loss_fct = SpanBoundarySmoothKLDivLoss(
    epsilon=EPSILON,  # 对应 KL Loss 的 epsilon
    D_smooth=D_SMOOTH,  # 对应 KL Loss 的 D_smooth
    reduction="mean",
    span_type="target"
)
argument_span_kl_loss_fct = SpanBoundarySmoothKLDivLoss(
    epsilon=EPSILON,
    D_smooth=D_SMOOTH,
    reduction="mean",
    span_type="argument"
)

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

# ===== Diversity Loss Function =====
def calculate_diversity_loss(
    per_sample_matched_pred_span_probs: List[List[Dict[str, torch.Tensor]]], 
    per_sample_matched_gt_indices: List[List[int]], 
    device: torch.device
) -> torch.Tensor:
    """
    Calculates diversity loss based on similarity of predicted span probability distributions
    for predictions that map to different ground truths.

    Args:
        per_sample_matched_pred_span_probs: List (batch) of lists (matched preds per sample) 
                                            of dicts. Each dict has keys 
                                            {'ts_probs', 'te_probs', 'as_probs', 'ae_probs'} 
                                            with probability tensor values (seq_len).
        per_sample_matched_gt_indices: List (batch) of lists (matched preds per sample) of int.
                                       Each int is the gt_idx for the corresponding prediction.
        device: Current torch device.
    Returns:
        A scalar tensor representing the diversity loss.
    """
    total_diversity_loss = torch.tensor(0.0, device=device)
    num_samples_with_diversity_pairs = 0

    batch_size = len(per_sample_matched_pred_span_probs)

    for b in range(batch_size):
        sample_pred_probs_list = per_sample_matched_pred_span_probs[b]
        sample_gt_indices_list = per_sample_matched_gt_indices[b]

        num_matched_preds_in_sample = len(sample_pred_probs_list)

        if num_matched_preds_in_sample < 2:
            continue
        
        # Only count this sample if it has potential for diversity pairs
        # This flag will be set true if at least one valid diversity pair is found
        sample_contributed_to_loss = False 
        sample_diversity_loss = torch.tensor(0.0, device=device)
        num_diversity_pairs_in_sample = 0

        for k in range(num_matched_preds_in_sample):
            for l in range(k + 1, num_matched_preds_in_sample):
                gt_idx_A = sample_gt_indices_list[k]
                gt_idx_B = sample_gt_indices_list[l]

                if gt_idx_A != gt_idx_B:
                    pred_A_probs_dict = sample_pred_probs_list[k]
                    pred_B_probs_dict = sample_pred_probs_list[l]

                    # Target Start Probs Similarity
                    similarity_ts = torch.sum(pred_A_probs_dict['ts_probs'] * pred_B_probs_dict['ts_probs'])
                    sample_diversity_loss += similarity_ts
                    
                    # Target End Probs Similarity
                    similarity_te = torch.sum(pred_A_probs_dict['te_probs'] * pred_B_probs_dict['te_probs'])
                    sample_diversity_loss += similarity_te

                    # Argument Start Probs Similarity
                    similarity_as = torch.sum(pred_A_probs_dict['as_probs'] * pred_B_probs_dict['as_probs'])
                    sample_diversity_loss += similarity_as

                    # Argument End Probs Similarity
                    similarity_ae = torch.sum(pred_A_probs_dict['ae_probs'] * pred_B_probs_dict['ae_probs'])
                    sample_diversity_loss += similarity_ae
                    
                    num_diversity_pairs_in_sample += 1
                    if not sample_contributed_to_loss:
                        sample_contributed_to_loss = True

        if num_diversity_pairs_in_sample > 0:
            total_diversity_loss += (sample_diversity_loss / num_diversity_pairs_in_sample)
        
        if sample_contributed_to_loss: # if we found any pair A,B where gt_idx_A != gt_idx_B
             num_samples_with_diversity_pairs +=1


    if num_samples_with_diversity_pairs > 0:
        return total_diversity_loss / num_samples_with_diversity_pairs
    else:
        return torch.tensor(0.0, device=device)


# ===== 4. compute_total_loss 的完整实现 =====
def compute_total_loss(outputs: Dict[str, torch.Tensor],
                       quads_labels_batch: List[List[Dict[str, Any]]],
                       model: nn.Module,
                       device: torch.device,
                       span_weight: float = 1.0,
                       group_weight: float = 0.5,
                       hateful_weight: float = 0.5,
                       biaffine_weight: float = 1.0,
                       diversity_loss_weight: float = 0.1,  # 新增参数：多样性损失的权重
                       iou_loss_ratio: float = 0.9,  # IoU 加权 BCE 的权重
                       kl_loss_ratio: float = 0.1  # 边界平滑 KL 的权重
                       ) -> (torch.Tensor, Dict[str, float]):
    # 1. 从模型输出里拿到各 logits
    # shape: [B, L, 1] → squeeze → [B, L]
    # 初始化各项损失为 Tensor
    # 1. 从模型输出里拿到各 logits
    t_start_logits = outputs['target_start_logits'].squeeze(-1)  # [B, L]
    t_end_logits = outputs['target_end_logits'].squeeze(-1)  # [B, L]
    a_start_logits = outputs['argument_start_logits'].squeeze(-1)  # [B, L]
    a_end_logits = outputs['argument_end_logits'].squeeze(-1)  # [B, L]
    sequence_output = outputs['sequence_output']  # [B, L, H]
    cls_output = outputs['cls_output']  # [B,   H]

    batch_size, seq_len = t_start_logits.size()

    # Data collection for diversity loss
    batch_span_probs_for_diversity = []
    batch_gt_indices_for_diversity = []

    for b_idx in range(batch_size):
        current_sample_quads = quads_labels_batch[b_idx]
        sample_span_probs_for_diversity = []
        sample_gt_indices_for_diversity = []

        if len(current_sample_quads) > 1:
            # These are global probs for sample b_idx, shared across its GT quads for diversity input
            sample_target_start_probs = torch.sigmoid(t_start_logits[b_idx])
            sample_target_end_probs = torch.sigmoid(t_end_logits[b_idx])
            sample_argument_start_probs = torch.sigmoid(a_start_logits[b_idx])
            sample_argument_end_probs = torch.sigmoid(a_end_logits[b_idx])

            for k_gt_idx, _ in enumerate(current_sample_quads):
                current_pred_span_probs = {
                    'ts_probs': sample_target_start_probs,
                    'te_probs': sample_target_end_probs,
                    'as_probs': sample_argument_start_probs,
                    'ae_probs': sample_argument_end_probs
                }
                sample_span_probs_for_diversity.append(current_pred_span_probs)
                sample_gt_indices_for_diversity.append(k_gt_idx)
        
        batch_span_probs_for_diversity.append(sample_span_probs_for_diversity)
        batch_gt_indices_for_diversity.append(sample_gt_indices_for_diversity)

    # ---- 2. Span Extraction Loss (IoU-Weighted BCE + Boundary-Smooth KL) ----
    # 2a. 计算 IoU 加权 BCE Loss
    target_span_iou_loss_component = target_span_iou_loss_fct(
        t_start_logits, t_end_logits, quads_labels_batch, device
    )
    argument_span_iou_loss_component = argument_span_iou_loss_fct(
        a_start_logits, a_end_logits, quads_labels_batch, device
    )
    total_span_iou_loss_component = target_span_iou_loss_component + argument_span_iou_loss_component

    # 2b. 计算边界平滑 KL 散度 Loss
    target_span_kl_loss_component = target_span_kl_loss_fct(
        t_start_logits, t_end_logits, quads_labels_batch, device
    )
    argument_span_kl_loss_component = argument_span_kl_loss_fct(
        a_start_logits, a_end_logits, quads_labels_batch, device
    )
    total_span_kl_loss_component = target_span_kl_loss_component + argument_span_kl_loss_component

    # 将两种 Span Loss 组件加权求和
    total_span_loss = (
            iou_loss_ratio * total_span_iou_loss_component +
            kl_loss_ratio * total_span_kl_loss_component
    )

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

    # ---- 5. Diversity Loss Calculation ----
    diversity_loss_val = torch.tensor(0.0, device=device) # Initialize
    if ENABLE_DIVERSITY_LOSS: # Use imported constant
        # Data collection for diversity loss is already done above
        diversity_loss_val = calculate_diversity_loss(
            batch_span_probs_for_diversity,
            batch_gt_indices_for_diversity,
            device
        )

    # ---- 6. 加权合并 ----
    loss = (
            span_weight * total_span_loss +
            biaffine_weight * total_pair_loss +
            group_weight * total_group_loss +
            hateful_weight * total_hateful_loss
    )
    if ENABLE_DIVERSITY_LOSS: # Use imported constant
        loss += diversity_loss_val * diversity_loss_weight # Use imported constant

    # 准备返回的各组件 Loss（转为 Python float 方便打印）
    loss_components = {
        'span_loss': total_span_loss.item(),  # 包含 IoU 和 KL 两个组件
        'iou_span_loss': total_span_iou_loss_component.item(),  # 新增：方便查看 IoU 部分损失
        'kl_span_loss': total_span_kl_loss_component.item(),  # 新增：方便查看 KL 部分损失
        'biaffine_loss': total_pair_loss.item(),
        'group_loss': total_group_loss.item(),
        'hateful_loss': total_hateful_loss.item(),
        'diversity_loss': diversity_loss_val.item() # Add diversity loss component (will be 0.0 if not enabled)
    }
    return loss, loss_components
