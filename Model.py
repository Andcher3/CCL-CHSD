# model.py
import torch
import torch.nn as nn
from transformers import BertModel
from Biaffine import BiaffinePairing  # 导入Biaffine模块
from typing import List, Dict, Any

# 假设 Hype.py 已经导入，并且 TARGET_GROUP_CLASS_NAME 和 MAX_SEQ_LENGTH 已定义
from Hype import TARGET_GROUP_CLASS_NAME, MAX_SEQ_LENGTH


class HateSpeechDetectionModel(nn.Module):
    # BIO Tags
    O_TAG = 0
    B_T_TAG = 1  # Begin-Target
    I_T_TAG = 2  # Inside-Target
    B_A_TAG = 3  # Begin-Argument
    I_A_TAG = 4  # Inside-Argument
    NUM_BIO_TAGS = 5

    def __init__(self, bert_model_name='bert-base-chinese'):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.hidden_size = self.bert.config.hidden_size  # BERT隐藏层维度

        # --- BIO Span Prediction Head ---
        self.bio_tag_head = nn.Linear(self.hidden_size, self.NUM_BIO_TAGS)

        # Biaffine Pairing Layer
        # Span representations will be self.hidden_size * 3 (start, end, mean-pooled content)
        self.biaffine_pairing = BiaffinePairing(input_dim=self.hidden_size * 3)

        # --- Classification Heads ---
        # Input: Target_vec (H*3) + Argument_vec (H*3) + CLS_vec (H) = H*7 (approx)
        classification_input_dim = (self.hidden_size * 3) + (self.hidden_size * 3) + self.hidden_size

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

        # 2. BIO Tag Logits Prediction
        # Shape: (batch_size, sequence_length, NUM_BIO_TAGS)
        bio_tag_logits = self.bio_tag_head(sequence_output)

        # Biaffine pairing and classification will still use _get_span_representation,
        # which itself uses sequence_output. Spans for Biaffine/classification will
        # need to be derived from BIO tags during inference, or from gold labels during training.
        # The current _get_span_representation is compatible with sequence_output.

        return {
            'bio_tag_logits': bio_tag_logits,
            'sequence_output': sequence_output,
            'cls_output': cls_output,
            'attention_mask': attention_mask
        }

    def _get_span_representation(self, sequence_output: torch.Tensor, start_idx: int, end_idx: int, batch_idx: int):
        """
        Helper to get span representation by concatenating start, end, and mean-pooled content token embeddings.
        """
        # Check for invalid indices or indices out of bounds
        # sequence_output shape: [batch_size, seq_len, hidden_size]
        seq_len = sequence_output.shape[1]
        if start_idx is None or end_idx is None or \
                start_idx == -1 or end_idx == -1 or \
                batch_idx < 0 or batch_idx >= sequence_output.shape[0] or \
                start_idx < 0 or start_idx >= seq_len or \
                end_idx < 0 or end_idx >= seq_len or \
                start_idx > end_idx: # Invalid span
            return torch.zeros(self.hidden_size * 3, device=sequence_output.device)

        start_embedding = sequence_output[batch_idx, start_idx]
        end_embedding = sequence_output[batch_idx, end_idx]

        mean_pooled_content: torch.Tensor
        if start_idx < end_idx -1 : # Content window has at least one token (e.g. start, content, end)
            # Content tokens are between start_idx (exclusive) and end_idx (exclusive)
            content_embeddings = sequence_output[batch_idx, start_idx + 1 : end_idx]
            if content_embeddings.nelement() == 0: # Should not happen if start_idx < end_idx - 1
                 mean_pooled_content = torch.zeros_like(start_embedding)
            else:
                mean_pooled_content = torch.mean(content_embeddings, dim=0)
        elif start_idx == end_idx : # Span of length 1 (just the start token)
            # No content tokens, or could argue the token itself is the content.
            # For consistency with start/end/pool, let's use zero for pool if no distinct content.
            mean_pooled_content = torch.zeros_like(start_embedding)
        elif start_idx == end_idx - 1: # Span of length 2 (start, end), no tokens strictly between them.
            mean_pooled_content = torch.zeros_like(start_embedding)
        else: # This case should ideally be caught by start_idx > end_idx above.
              # If somehow reached (e.g. logic error), provide zeros.
            mean_pooled_content = torch.zeros_like(start_embedding)


        return torch.cat((start_embedding, end_embedding, mean_pooled_content), dim=-1)

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