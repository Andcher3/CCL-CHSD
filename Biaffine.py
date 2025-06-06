import torch
import torch.nn as nn


class BiaffinePairing(nn.Module):
    def __init__(self, input_dim): # Changed hidden_size to input_dim
        super().__init__()
        # W: [input_dim, input_dim]
        self.W = nn.Parameter(torch.randn(input_dim, input_dim) * 0.02)
        # U: [2*input_dim, 1]
        self.U = nn.Parameter(torch.randn(2 * input_dim, 1) * 0.02)
        # Bias: [1]
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, target_spans: torch.Tensor, argument_spans: torch.Tensor):
        """
        参数：
        - target_spans: Tensor of shape [n, H] (n = num_target_spans)
        - argument_spans: Tensor of shape [m, H] (m = num_argument_spans)
        返回：
        - scores: Tensor of shape [n, m]，每个元素是 target i 与 argument j 的配对得分（未经过 Sigmoid）
        """
        # 先计算 T W A^T:  [n, H] × [H, H] = [n, H]；再与 [m, H]^T → [n, m]
        # TW: [n, H]
        TW = target_spans @ self.W  # 矩阵乘积
        # TW ([n,H]) 与 argument_spans^T ([H,m]) → [n,m]
        biaffine_term = TW @ argument_spans.transpose(0, 1)  # [n, m]

        # 再计算 U^T [T_i; A_j] 这部分
        # 先把 target_spans 扩展成 [n, 1, H]，argument_spans 扩展成 [1, m, H]
        # 再拼成 [n, m, 2H]，最后 matmul U → [n, m, 1] → squeeze → [n, m]
        n, H = target_spans.size() # H is now input_dim
        m, _ = argument_spans.size() # Argument spans also have H (input_dim)
        # expand
        T_exp = target_spans.unsqueeze(1).expand(n, m, H)       # [n, m, H]
        A_exp = argument_spans.unsqueeze(0).expand(n, m, H)     # [n, m, H]
        all_concat = torch.cat([T_exp, A_exp], dim=-1)          # [n, m, 2*H]
        # U: [2*H, 1] (where H is input_dim for U's definition)
        # self.U is [2*input_dim, 1]. H from target_spans.size() is input_dim. So 2*H is correct.
        U_reshaped = self.U.view(1, 1, 2 * H, 1)
        # all_concat: [n, m, 2*H] → [n, m, 1] after matmul
        pairwise_term = torch.matmul(all_concat.unsqueeze(-2), U_reshaped).squeeze(-1).squeeze(-1)  # [n, m]

        # 最后得分矩阵 = biaffine_term + pairwise_term + b
        scores = biaffine_term + pairwise_term + self.b  # [n, m]
        return scores


if __name__ == '__main__':
    biaffine = BiaffinePairing(input_dim=16) # Test with input_dim
    # print(biaffine.get_parameter) # Assuming get_parameter was a placeholder/error
    print(biaffine.W.shape)
    print(biaffine.U.shape)