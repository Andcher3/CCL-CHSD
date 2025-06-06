# model.py
import torch
import torch.nn as nn
from transformers import BertModel
from Biaffine import BiaffinePairing  # 导入Biaffine模块
from typing import List, Dict, Any

# 假设 Hype.py 已经导入，并且 TARGET_GROUP_CLASS_NAME 和 MAX_SEQ_LENGTH 已定义
from Hype import TARGET_GROUP_CLASS_NAME, MAX_SEQ_LENGTH


class HateSpeechDetectionModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-chinese'):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.hidden_size = self.bert.config.hidden_size  # BERT隐藏层维度

        # self.span_bilstm removed for max-pooling based representation

        # --- Span Extraction Heads ---
        # 预测Target Span的起始和结束位置
        self.target_start_head = nn.Linear(self.hidden_size, 1)
        self.target_end_head = nn.Linear(self.hidden_size, 1)

        # 预测Argument Span的起始和结束位置
        self.argument_start_head = nn.Linear(self.hidden_size, 1)
        self.argument_end_head = nn.Linear(self.hidden_size, 1)

        # Biaffine Pairing Layer
        # Span representations are now self.hidden_size (output of BiLSTM)
        self.biaffine_pairing = BiaffinePairing(input_dim=self.hidden_size)

        # --- Classification Heads ---
        # Input: Target_vec (H) + Argument_vec (H) + CLS_vec (H) = H*3
        classification_input_dim = self.hidden_size * 3

        # Targeted Group分类头
        self.group_classifier = nn.Sequential(
            nn.Linear(classification_input_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, len(TARGET_GROUP_CLASS_NAME))
        )

        # Hateful分类头
        self.hateful_classifier = nn.Sequential(
            nn.Linear(classification_input_dim, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size // 2, 1)
        )

    def forward(self, input_ids, attention_mask, token_type_ids,
                # 在训练时，可能需要传入真实的四元组标签来计算损失
                # 但在forward方法中，我们主要关注预测输出
                # biaffine_labels=None, # 假设用于Biaffine的标签，如果实现Biaffine
                # true_quads=None # 真实的四元组，用于匹配和损失计算
                ):
        # 1. BERT 编码
        # output.last_hidden_state: (batch_size, sequence_length, hidden_size)
        # output.pooler_output: (batch_size, hidden_size) # [CLS] token经过池化层
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

        sequence_output = outputs.last_hidden_state
        cls_output = outputs.pooler_output  # 使用BERT的[CLS]输出作为句子级表示

        # 2. Span Extraction Logits
        # (batch_size, sequence_length, 1)
        target_start_logits = self.target_start_head(sequence_output)
        target_end_logits = self.target_end_head(sequence_output)
        argument_start_logits = self.argument_start_head(sequence_output)
        argument_end_logits = self.argument_end_head(sequence_output)

        # 在forward中直接进行Biaffine配对可能比较复杂，通常在损失函数计算时处理
        # 或者，更常见的是，先预测所有可能的span，然后在推理时再进行配对
        # 或者，对于训练，我们可以根据真实的quads来获取对应的span表示进行分类

        # 对于训练和推理，我们都需要一种方式来获取 (Target Span, Argument Span) 的联合表示
        # 假设我们已经有了预测或真实的 (t_start_token, t_end_token, a_start_token, a_end_token)
        # 这一部分将在实际的训练循环或推理函数中处理，而不是在forward中直接完成
        # 因为forward函数主要负责从输入到所有原始预测 logits 的计算。

        # 返回原始的 logits，后续在训练循环或推理函数中进行处理
        return {
            'target_start_logits': target_start_logits,
            'target_end_logits': target_end_logits,
            'argument_start_logits': argument_start_logits,
            'argument_end_logits': argument_end_logits,
            'sequence_output': sequence_output,  # 返回序列输出，以便后续获取Span表示
            'cls_output': cls_output  # 返回CLS输出，以便后续获取句子级表示
        }

    def _get_span_representation(self, sequence_output: torch.Tensor, start_idx: int, end_idx: int, batch_idx: int):
        """
        Helper to get span representation using max-pooling over token embeddings.
        Output dimension will be self.hidden_size.
        """
        # Check for invalid indices or indices out of bounds
        seq_len = sequence_output.shape[1]
        if start_idx is None or end_idx is None or \
                start_idx == -1 or end_idx == -1 or \
                batch_idx < 0 or batch_idx >= sequence_output.shape[0] or \
                start_idx < 0 or start_idx >= seq_len or \
                end_idx < 0 or end_idx >= seq_len:
            return torch.zeros(self.hidden_size, device=sequence_output.device)

        if start_idx > end_idx: # Invalid span order
            return torch.zeros(self.hidden_size, device=sequence_output.device)

        # Extract token embeddings for the span
        # Shape: (span_length, bert_hidden_size)
        span_embeddings = sequence_output[batch_idx, start_idx : end_idx + 1, :]

        # Check if the span_embeddings tensor is empty (e.g. if start_idx > end_idx resulted in an empty slice,
        # or if start_idx == end_idx + 1 which means no tokens in span)
        # span_embeddings.shape[0] would be the span_length.
        if span_embeddings.shape[0] == 0:
            return torch.zeros(self.hidden_size, device=sequence_output.device)

        # Apply max-pooling over the token embeddings within the span.
        # torch.max returns (values, indices). We only need values.
        # The pooling is done over dim=0 (the sequence/span_length dimension).
        # Output shape: (bert_hidden_size)
        pooled_representation = torch.max(span_embeddings, dim=0)[0]

        return pooled_representation

    def classify_quad(self, sequence_output: torch.Tensor, cls_output: torch.Tensor,
                      t_start_token: int, t_end_token: int,
                      a_start_token: int, a_end_token: int, batch_idx: int):
        """
        Takes the BERT outputs and token-level span indices for a single quad within a batch,
        and performs classification for group and hateful.
        This method is meant to be called *after* spans have been identified/paired.
        """
        target_vec = self._get_span_representation(sequence_output, t_start_token, t_end_token, batch_idx)
        argument_vec = self._get_span_representation(sequence_output, a_start_token, a_end_token, batch_idx)

        # Concatenate target, argument, and CLS representations
        combined_vec = torch.cat([target_vec, argument_vec, cls_output[batch_idx]], dim=-1)

        group_logits = self.group_classifier(combined_vec)
        hateful_logits = self.hateful_classifier(combined_vec)

        return group_logits, hateful_logits


if __name__ == '__main__':
    model = HateSpeechDetectionModel(bert_model_name='bert-base-chinese')