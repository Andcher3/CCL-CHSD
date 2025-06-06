import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any


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
                end_logits:   torch.Tensor,  # [B, L]
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
        end_labels   = torch.zeros((B, L), device=device)
        # 记录 gold spans 列表，便于后续 IoU 计算
        gold_spans_for_iou = [[] for _ in range(B)] # Renamed from gold_spans_start

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
                if 0 <= s <= e < L: # Use s, e based on span_type
                    gold_spans_for_iou[b].append((s, e))

        # 1. 计算 token-level 基础 BCEWithLogitsLoss
        bce_loss_start = F.binary_cross_entropy_with_logits(start_logits, start_labels, reduction="none")
        bce_loss_end   = F.binary_cross_entropy_with_logits(end_logits,   end_labels,   reduction="none")

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
                    for (s_gold, e_gold) in spans: # Use s_gold, e_gold from gold_spans_for_iou
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
                    for (s_gold, e_gold) in spans: # Use s_gold, e_gold
                        # 这里 j 视作“span (j,j)”
                        inter = 1 if (s_gold <= j <= e_gold) else 0
                        union = (e_gold - s_gold + 1) + 1 - inter
                        iou_j = inter / union if union > 0 else 0.0
                        if iou_j > max_iou_j:
                            max_iou_j = iou_j
                    weight_end[b, j] = self.alpha * ((1 + max_iou_j) ** self.gamma)

        # 4. 最终加权 BCE Loss
        loss_start = (weight_start * bce_loss_start).mean()
        loss_end   = (weight_end   * bce_loss_end).mean()

        total_loss = loss_start + loss_end  # 这里只包含 start/end 部分，后面还可以加 argument、biaffine、分类等

        return total_loss


