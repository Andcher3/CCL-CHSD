import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any
from Hype import *


class SpanIoUWeightedLoss(nn.Module):
    """
    把 IoU 思想引入到 'token-level start/end 二分类' 的动态加权损失里
    适用于：每个句子可能有多个四元组（多个正例 start/end），
    我们对“token i 是正例 start” 或 “token j 是正例 end” 做二分类，
    对那些“负样本 token 与任意正例 span 有较大 overlap”的位置加大权重。
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction="mean", span_type: str = "target"):
        """
        alpha: 正例 vs 负例的基准平衡因子。通常 0.1~0.25 左右。
        gamma: Focal 中调节“难易”程度的参数。这里也可用做平滑指数。
        span_type: "target" or "argument", to determine which keys to use from quads_labels.
        reduction: 'mean' 或 'sum' 用于最终返回标量。
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.span_type = span_type

    def forward(self,
                start_logits: torch.Tensor,  # [B, L]
                end_logits: torch.Tensor,  # [B, L]
                quads_labels: List[List[Dict[str, Any]]],
                device: torch.device) -> torch.Tensor:
        """
        start_logits, end_logits:都是未经 sigmoid 的 raw logits，形状 [B, L]。
        quads_labels: 每个 batch 元素里包含多个 quad_dict，quad_dict 包含
                      't_start_token', 't_end_token', 'a_start_token', 'a_end_token'
                      以及'group_vector','hateful_flag'等。
        """
        B, L = start_logits.size()

        # 对每个 token 设置正负标签
        start_labels = torch.zeros((B, L), device=device)
        end_labels = torch.zeros((B, L), device=device)
        # 记录 gold spans 列表，便于后续 IoU 计算
        gold_spans_for_iou = [[] for _ in range(B)]  # Renamed from gold_spans_start

        for b, quads_list in enumerate(quads_labels):
            for quad in quads_list:
                if self.span_type == "target":
                    s, e = quad['t_start_token'].item(), quad['t_end_token'].item()
                elif self.span_type == "argument":
                    s, e = quad['a_start_token'].item(), quad['a_end_token'].item()
                else:
                    raise ValueError(f"Invalid span_type: {self.span_type} in SpanIoUWeightedLoss")

                if 0 <= s < L:
                    start_labels[b, s] = 1.0
                if 0 <= e < L:
                    end_labels[b, e] = 1.0
                # 记录 gold span 对（后面算 IoU 用）
                if 0 <= s <= e < L:  # Use s, e based on span_type
                    gold_spans_for_iou[b].append((s, e))

        # 1. 计算 token-level 基础 BCEWithLogitsLoss
        bce_loss_start = F.binary_cross_entropy_with_logits(start_logits, start_labels, reduction="none")
        bce_loss_end = F.binary_cross_entropy_with_logits(end_logits, end_labels, reduction="none")

        # 2. 对“负样本 token”计算 IoU-based 加权系数
        #  例如：若 token i 本身不是任何 span 的 start（start_labels[b,i]=0），
        #      我们依次判断 i 落在哪些 gold_spans_start[b] 里，并取最大 IoU。
        #  最终权重 = alpha * (1 + maxIoU(i))，若 token i 是正例，则权重 = 1 - alpha。
        #  这里把 gamma 项也加进来做调节：
        #    weight_start[b,i] = (1 - alpha)   if start_labels[b,i] == 1
        #                      = alpha * (1 + maxIoU(i))^gamma   otherwise
        weight_start = torch.zeros((B, L), device=device)
        for b in range(B):
            spans = gold_spans_for_iou[b]  # 该样本的所有 gold span (s, e)
            for i in range(L):
                if start_labels[b, i] == 1:  # 正例
                    weight_start[b, i] = (1 - self.alpha)
                else:  # 负例 token，需要算 IoU
                    max_iou_i = 0.0
                    for (s_gold, e_gold) in spans:  # Use s_gold, e_gold from gold_spans_for_iou
                        # 这里 i 视作“span (i,i) 只含一个 token”
                        inter = 1 if (s_gold <= i <= e_gold) else 0
                        # span_len_gold = e_gold - s_gold + 1, span_len_pred = 1
                        union = (e_gold - s_gold + 1) + 1 - inter
                        iou_i = inter / union if union > 0 else 0.0
                        if iou_i > max_iou_i:
                            max_iou_i = iou_i
                    weight_start[b, i] = self.alpha * ((1 + max_iou_i) ** self.gamma)

        # 3. end 部分同理
        weight_end = torch.zeros((B, L), device=device)
        for b in range(B):
            spans = gold_spans_for_iou[b]  # Use gold_spans_for_iou, which is specific to target/argument
            for j in range(L):
                if end_labels[b, j] == 1:
                    weight_end[b, j] = (1 - self.alpha)
                else:
                    max_iou_j = 0.0
                    for (s_gold, e_gold) in spans:  # Use s_gold, e_gold
                        # 这里 j 视作“span (j,j)”
                        inter = 1 if (s_gold <= j <= e_gold) else 0
                        union = (e_gold - s_gold + 1) + 1 - inter
                        iou_j = inter / union if union > 0 else 0.0
                        if iou_j > max_iou_j:
                            max_iou_j = iou_j
                    weight_end[b, j] = self.alpha * ((1 + max_iou_j) ** self.gamma)

        # 4. 最终加权 BCE Loss
        loss_start = (weight_start * bce_loss_start).mean()
        loss_end = (weight_end * bce_loss_end).mean()

        total_loss = loss_start + loss_end  # 这里只包含 start/end 部分，后面还可以加 argument、biaffine、分类等

        return total_loss


class SpanBoundarySmoothKLDivLoss(nn.Module):
    """
    一个 Span-level 的 KL 散度损失，用于鼓励模型在真实 Span 边界附近形成平滑预测。
    它将所有可能的候选 Span 作为多分类的类别，并为每个真实 Span 构建一个平滑目标分布。
    通过 Top-K 筛选和强制包含真实 Span 来减少计算量。
    """

    def __init__(self, epsilon: float = 0.1, D_smooth: int = 1, reduction: str = "mean", span_type: str = "target"):
        super().__init__()
        self.epsilon = epsilon
        self.D_smooth = D_smooth
        self.reduction = reduction
        self.span_type = span_type

    def forward(self,
                start_logits: torch.Tensor,  # [B, L]
                end_logits:   torch.Tensor,  # [B, L]
                quads_labels_batch: List[List[Dict[str, Any]]],
                device: torch.device) -> torch.Tensor:
        B, L = start_logits.size()
        accumulated_kl_loss = torch.tensor(0.0, device=device)
        num_valid_gold_spans_processed = 0

        # 获取 Hype 中定义的 TOPK_SPAN (例如 8 或 10)
        k_span_for_comb = TOPK_SPAN # 控制用于组合的 Top-K start/end 数量

        for b in range(B): # 遍历批次中的每个样本
            current_start_logits = start_logits[b]  # [L]
            current_end_logits = end_logits[b]      # [L]
            current_quads_list = quads_labels_batch[b]

            current_gold_spans_for_kl = []
            for quad in current_quads_list:
                s, e = -1, -1
                if self.span_type == "target":
                    s, e = quad['t_start_token'].item(), quad['t_end_token'].item()
                elif self.span_type == "argument":
                    s, e = quad['a_start_token'].item(), quad['a_end_token'].item()
                
                if 0 <= s <= e < L:
                    current_gold_spans_for_kl.append((s, e))
            
            if not current_gold_spans_for_kl and len(current_quads_list) > 0: # 如果有quads但gold span无效，继续
                continue 
            elif not current_gold_spans_for_kl and len(current_quads_list) == 0: # 没有quads, 跳过
                continue


            # 2. 生成 Top-K 候选 Span (优化点)
            # 在有效区间 [1, L-2] 内获取 Top-K
            valid_start_idx_for_cand = 1
            valid_end_idx_for_cand = L - 2 
            
            if valid_end_idx_for_cand < valid_start_idx_for_cand: # 文本太短
                continue

            # 确保 k_span_for_comb 不会大于实际可用的 token 数量
            actual_k_for_topk = min(k_span_for_comb, valid_end_idx_for_cand - valid_start_idx_for_cand + 1)
            if actual_k_for_topk <= 0: # 没法取 Top-K
                continue

            # 在 GPU 上获取 Top-K 的 start/end 索引
            topk_s_indices = torch.topk(current_start_logits[valid_start_idx_for_cand : valid_end_idx_for_cand + 1], k=actual_k_for_topk).indices + valid_start_idx_for_cand
            topk_e_indices = torch.topk(current_end_logits[valid_start_idx_for_cand : valid_end_idx_for_cand + 1], k=actual_k_for_topk).indices + valid_start_idx_for_cand

            all_raw_candidate_spans_with_scores = [] 
            # 只组合 Top-K 的索引
            for s_idx in topk_s_indices.tolist():
                for e_idx in topk_e_indices.tolist():
                    if s_idx <= e_idx and (e_idx - s_idx + 1) <= MAX_SPAN_LENGTH:
                        score = current_start_logits[s_idx] + current_end_logits[e_idx] # Sum of logits
                        all_raw_candidate_spans_with_scores.append(((s_idx, e_idx), score))
            
            if not all_raw_candidate_spans_with_scores:
                continue
            # 3. 筛选候选 Span (Top-K + 强制包含真实 Span)
            # 先按得分排序
            all_raw_candidate_spans_with_scores.sort(key=lambda x: x[1], reverse=True)

            # 初始化最终候选集
            final_candidate_spans_kl = []  # List of (s, e) tuples
            final_candidate_raw_scores_kl = []  # List of corresponding raw scores

            # 使用集合来快速检查 Span 是否已存在
            added_spans_set = set()
            # print(4)
            # 强制添加所有真实 Span
            for gs, ge in current_gold_spans_for_kl:
                if (gs, ge) not in added_spans_set:
                    final_candidate_spans_kl.append((gs, ge))
                    # 获取真实 Span 的预测得分 (如果它存在于原始 logits 区域)
                    score = current_start_logits[gs] + current_end_logits[ge]
                    final_candidate_raw_scores_kl.append(score)
                    added_spans_set.add((gs, ge))

            # 添加 Top-K 预测的 Span (排除已添加的真实 Span)
            # 这里 TOPK_SPAN 可以用来控制从所有候选里选多少个 Top-K，作为除了真实 Span 之外的额外候选
            # 我们要确保总数不超过 MAX_CANDIDATE_SPANS_PER_SAMPLE_FOR_KL_LOSS

            # 计算还需要添加多少个非真实 Span 的候选
            num_to_add_from_topk = MAX_CANDIDATE_SPANS_PER_SAMPLE_FOR_KL_LOSS - len(final_candidate_spans_kl)
            # print(5)
            if num_to_add_from_topk > 0:
                for (s, e), score in all_raw_candidate_spans_with_scores:
                    if (s, e) not in added_spans_set:
                        final_candidate_spans_kl.append((s, e))
                        final_candidate_raw_scores_kl.append(score)
                        added_spans_set.add((s, e))
                        num_to_add_from_topk -= 1
                        if num_to_add_from_topk == 0:
                            break

            # 如果最终没有候选 Span (可能因为没有真实 Span 或没有有效预测 Span)
            if not final_candidate_spans_kl:
                continue

            # 将最终候选 Span 的原始得分转换为 log 概率分布
            final_candidate_scores_tensor = torch.stack(final_candidate_raw_scores_kl)
            pred_log_probs = F.log_softmax(final_candidate_scores_tensor, dim=0)
            # print(6)
            # 4. 遍历真实 Span，构建平滑目标分布并计算 KL 损失
            for gold_s, gold_e in current_gold_spans_for_kl:
                # 找到当前真实 Span 在最终候选 Span 列表中的索引
                try:
                    gold_span_candidate_idx = final_candidate_spans_kl.index((gold_s, gold_e))
                except ValueError:
                    # 如果真实 Span 由于某种原因没能进入最终候选列表，则跳过此 gold span 的损失计算
                    # (这通常不应该发生，因为我们强制添加了)
                    continue

                num_final_candidates = len(final_candidate_spans_kl)
                target_dist = torch.zeros(num_final_candidates, device=device, dtype=torch.float)

                neighbors_indices = []
                for i, (cs, ce) in enumerate(final_candidate_spans_kl):
                    if i == gold_span_candidate_idx:
                        continue
                    manhattan_dist = abs(gold_s - cs) + abs(gold_e - ce)
                    if manhattan_dist <= self.D_smooth:
                        neighbors_indices.append(i)

                if not neighbors_indices:
                    target_dist[gold_span_candidate_idx] = 1.0
                else:
                    target_dist[gold_span_candidate_idx] = 1.0 - self.epsilon
                    if len(neighbors_indices) > 0:  # 确保分母不为0
                        prob_for_each_neighbor = self.epsilon / len(neighbors_indices)
                        for neighbor_idx in neighbors_indices:
                            target_dist[neighbor_idx] = prob_for_each_neighbor
                # print(7)
                # 归一化目标分布
                current_sum_target_dist = target_dist.sum().item()
                if current_sum_target_dist > 1e-6:  # 避免除以0
                    target_dist = target_dist / current_sum_target_dist
                else:
                    # 目标分布异常，跳过此 gold span 的损失计算
                    continue

                # 计算 KL 散度损失
                loss_one_gold_span = F.kl_div(pred_log_probs, target_dist, reduction='sum', log_target=False)

                if not torch.isnan(loss_one_gold_span) and not torch.isinf(loss_one_gold_span):
                    accumulated_kl_loss += loss_one_gold_span
                    num_valid_gold_spans_processed += 1
        # print(8)
        if num_valid_gold_spans_processed == 0:
            return torch.tensor(0.0, device=device)

        if self.reduction == "mean":
            return accumulated_kl_loss / num_valid_gold_spans_processed
        elif self.reduction == "sum":
            return accumulated_kl_loss
        else:
            raise ValueError(f"Unsupported reduction type: {self.reduction}")