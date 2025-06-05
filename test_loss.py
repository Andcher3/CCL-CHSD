# test_loss.py
import torch
import torch.nn as nn
from data_extractor import load_data  # 如果需要，可以忽略数据加载部分
from Hype import *  # 确保这两个常量在 Hype.py 中有定义
from loss import compute_total_loss
from Biaffine import BiaffinePairing
from typing import Dict, List, Any


# 先定义一个最小版的 HateSpeechDetectionModel，用于测试
class DummyModel(nn.Module):
    def __init__(self, hidden_size=16, num_groups=6):
        super().__init__()
        self.hidden_size = hidden_size
        # 伪造 4 个 span 抽取 head：我们直接在 test 中随机初始化 logits
        # 仅为 classify_quad 测试定义一个小的分类器
        self.group_classifier = nn.Linear(hidden_size * 3, num_groups)
        self.hateful_classifier = nn.Linear(hidden_size * 3, 1)
        self.biaffine_pairing = BiaffinePairing(hidden_size)

    def _get_span_representation(self, sequence_output: torch.Tensor,
                                 start_idx: int, end_idx: int, batch_idx: int):
        """
        对每个 span 进行简单求和池化，返回 (hidden_size,) 向量
        """
        if start_idx < 0 or end_idx < start_idx:
            return torch.zeros(self.hidden_size, device=sequence_output.device)
        span_vec = sequence_output[batch_idx, start_idx:end_idx + 1].mean(dim=0)
        return span_vec

    def classify_quad(self, sequence_output: torch.Tensor, cls_output: torch.Tensor,
                      t_start_token: int, t_end_token: int,
                      a_start_token: int, a_end_token: int, batch_idx: int):
        """
        拼接 [Target_vec; Argument_vec; CLS_vec] 给两个小 Linear 并返回
        """
        tgt_vec = self._get_span_representation(sequence_output, t_start_token, t_end_token, batch_idx)
        arg_vec = self._get_span_representation(sequence_output, a_start_token, a_end_token, batch_idx)
        cls_vec = cls_output[batch_idx]
        combined = torch.cat([tgt_vec, arg_vec, cls_vec], dim=-1)  # [3H]
        group_logits = self.group_classifier(combined)  # [num_groups]
        hateful_logits = self.hateful_classifier(combined)  # [1]
        return group_logits.unsqueeze(0), hateful_logits.unsqueeze(0)


def test_compute_loss():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 2
    seq_len = 8
    hidden_size = 16

    # 1. 随机构造 span logits & hidden states
    #   target_start_logits: [B, L, 1], ...
    t_start_logits = torch.randn(batch_size, seq_len, 1, device=device)
    t_end_logits = torch.randn(batch_size, seq_len, 1, device=device)
    a_start_logits = torch.randn(batch_size, seq_len, 1, device=device)
    a_end_logits = torch.randn(batch_size, seq_len, 1, device=device)

    # 随机隐藏态
    sequence_output = torch.randn(batch_size, seq_len, hidden_size, device=device)
    cls_output = torch.randn(batch_size, hidden_size, device=device)

    outputs: Dict[str, torch.Tensor] = {
        'target_start_logits': t_start_logits,
        'target_end_logits': t_end_logits,
        'argument_start_logits': a_start_logits,
        'argument_end_logits': a_end_logits,
        'sequence_output': sequence_output,
        'cls_output': cls_output
    }

    # 2. 构造 quads_labels_batch：每个样本给 1~2 条 gold 四元组
    #    例如 sample 0：target span = (1,2), argument span = (4,5)，
    #                 group_vector = [1,0,0,0,0,0], hateful_flag = 1
    quads_labels_batch: List[List[Dict[str, Any]]] = [
        [  # 样本 0
            {
                't_start_token': torch.tensor(1),  # 假设 gold target 从 token1 到 token2
                't_end_token': torch.tensor(2),
                'a_start_token': torch.tensor(4),
                'a_end_token': torch.tensor(5),
                'group_vector': torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.float),
                'hateful_flag': torch.tensor(1, dtype=torch.long)
            }
        ],
        [  # 样本 1，设定两个 gold 四元组
            {
                't_start_token': torch.tensor(0),
                't_end_token': torch.tensor(0),
                'a_start_token': torch.tensor(2),
                'a_end_token': torch.tensor(3),
                'group_vector': torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.float),
                'hateful_flag': torch.tensor(0, dtype=torch.long)
            },
            {
                't_start_token': torch.tensor(5),
                't_end_token': torch.tensor(5),
                'a_start_token': torch.tensor(6),
                'a_end_token': torch.tensor(7),
                'group_vector': torch.tensor([0, 0, 1, 0, 0, 0], dtype=torch.float),
                'hateful_flag': torch.tensor(1, dtype=torch.long)
            }
        ]
    ]

    # 3. 实例化 DummyModel，并放到 device
    model = DummyModel(hidden_size=hidden_size, num_groups=6).to(device)

    # 4. 调用 compute_total_loss
    loss, loss_components = compute_total_loss(
        outputs=outputs,
        quads_labels_batch=quads_labels_batch,
        model=model,
        device=device,
        span_weight=1.0,
        group_weight=1.0,
        hateful_weight=1.0,
        biaffine_weight=1.0  # 即便 Biaffine 计算出来全 0，也不影响流程
    )

    print("== 测试 compute_total_loss ==")
    print("Total loss:", loss.item())
    print("各项子 Loss：", loss_components)


if __name__ == '__main__':
    test_compute_loss()
