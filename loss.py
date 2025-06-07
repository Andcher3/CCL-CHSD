# loss.py
import torch
import torch.nn as nn
from typing import Dict, List, Any
# from iou_loss import SpanIoUWeightedLoss # Not used with BIO
from Hype import * # Assuming TOPK_SPAN, MAX_SPAN_LENGTH might be used by Biaffine still
import torch.nn.functional as F

# BIO Tag Definitions
O_TAG = 0
B_T_TAG = 1  # Begin-Target
I_T_TAG = 2  # Inside-Target
B_A_TAG = 3  # Begin-Argument
I_A_TAG = 4  # Inside-Argument
NUM_BIO_TAGS = 5
IGNORE_INDEX = -100  # PyTorch's CrossEntropyLoss default ignore_index

# Loss function for BIO span prediction
bio_span_loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

# Loss functions for other parts of the model (if they remain)
group_loss_fct = nn.BCEWithLogitsLoss(reduction='mean')
hateful_loss_fct = nn.BCEWithLogitsLoss(reduction='mean')
pair_loss_fct = nn.BCEWithLogitsLoss(reduction='mean')


# ===== 4. compute_total_loss 的完整实现 =====
def compute_total_loss(outputs: Dict[str, torch.Tensor],
                       quads_labels_batch: List[List[Dict[str, Any]]],
                       model: nn.Module, # model is passed for _get_span_representation
                       device: torch.device,
                       span_weight: float = 1.0,
                       group_weight: float = 0.5,
                       hateful_weight: float = 0.5,
                       biaffine_weight: float = 1.0
                       ) -> (torch.Tensor, Dict[str, float]):

    # 1. 从模型输出里拿到各 logits and other outputs
    bio_tag_logits = outputs['bio_tag_logits']  # Shape: (batch_size, seq_len, NUM_BIO_TAGS)
    attention_mask = outputs['attention_mask']  # Shape: (batch_size, seq_len)
    sequence_output = outputs['sequence_output'] # Shape: (batch_size, seq_len, hidden_size)
    cls_output = outputs['cls_output']          # Shape: (batch_size, hidden_size)

    batch_size, seq_len, _ = bio_tag_logits.shape

    # ---- 2. BIO Span Extraction Loss ----
    all_gold_bio_labels_for_batch = []
    for b in range(batch_size):
        gold_bio_labels_sample = torch.full((seq_len,), O_TAG, dtype=torch.long, device=device)

        # Apply IGNORE_INDEX to padded tokens (where attention_mask is 0)
        # Valid tokens (attention_mask == 1) will be processed for actual BIO tags.
        gold_bio_labels_sample[attention_mask[b] == 0] = IGNORE_INDEX

        for quad in quads_labels_batch[b]:
            # Target span
            ts = quad['t_start_token'].item()
            te = quad['t_end_token'].item()
            if ts != -1 and te != -1: # Ensure valid span from data
                # Further check if indices are within seq_len and not padded
                if 0 <= ts < seq_len and attention_mask[b, ts] == 1:
                    gold_bio_labels_sample[ts] = B_T_TAG
                    for k_idx in range(ts + 1, te + 1): # te is inclusive
                        if 0 <= k_idx < seq_len and attention_mask[b, k_idx] == 1:
                            gold_bio_labels_sample[k_idx] = I_T_TAG
                        # else: break # Span goes into padding or out of bounds

            # Argument span
            als = quad['a_start_token'].item()
            ale = quad['a_end_token'].item()
            if als != -1 and ale != -1: # Ensure valid span from data
                # Further check if indices are within seq_len and not padded
                if 0 <= als < seq_len and attention_mask[b, als] == 1:
                    # Prioritize Target tags: only write Argument tag if current is O or IGNORE_INDEX
                    # (IGNORE_INDEX check is mostly for safety, should be O_TAG for valid tokens)
                    if gold_bio_labels_sample[als] == O_TAG or gold_bio_labels_sample[als] == IGNORE_INDEX:
                        gold_bio_labels_sample[als] = B_A_TAG
                for k_idx in range(als + 1, ale + 1): # ale is inclusive
                    if 0 <= k_idx < seq_len and attention_mask[b, k_idx] == 1:
                        if gold_bio_labels_sample[k_idx] == O_TAG or gold_bio_labels_sample[k_idx] == IGNORE_INDEX:
                             gold_bio_labels_sample[k_idx] = I_A_TAG
                        # else: break # Span goes into padding or out of bounds / overlaps with higher priority tag
        all_gold_bio_labels_for_batch.append(gold_bio_labels_sample)

    gold_bio_labels = torch.stack(all_gold_bio_labels_for_batch, dim=0)  # Shape: (batch_size, seq_len)

    # Calculate BIO Span Loss
    # Reshape logits: (batch_size, seq_len, NUM_BIO_TAGS) -> (batch_size * seq_len, NUM_BIO_TAGS)
    # Reshape labels: (batch_size, seq_len) -> (batch_size * seq_len)
    reshaped_logits = bio_tag_logits.view(batch_size * seq_len, NUM_BIO_TAGS)
    reshaped_labels = gold_bio_labels.view(batch_size * seq_len)
    total_span_loss = bio_span_loss_fct(reshaped_logits, reshaped_labels)


    # ---- 3. Biaffine 配对 Loss ----
    # Spans for Biaffine must now be derived from gold labels for training.
    # In inference, they would be decoded from BIO tags.
    all_target_spans_gold = []  # Elements: (batch_idx, ts, te)
    all_argument_spans_gold = [] # Elements: (batch_idx, as, ae)

    for b, quads_list in enumerate(quads_labels_batch):
        for quad in quads_list:
            ts, te = quad['t_start_token'].item(), quad['t_end_token'].item()
            als, ale = quad['a_start_token'].item(), quad['a_end_token'].item()

            # Only add valid spans that are within sequence length and not padded
            # (though _get_span_representation also has checks)
            if ts != -1 and te != -1 and 0 <= ts <= te < seq_len:
                 all_target_spans_gold.append((b, ts, te))
            if als != -1 and ale != -1 and 0 <= als <= ale < seq_len:
                 all_argument_spans_gold.append((b, als, ale))

    # Deduplicate to prevent redundant computations if spans are shared across quads
    all_target_spans_gold = sorted(list(set(all_target_spans_gold)))
    all_argument_spans_gold = sorted(list(set(all_argument_spans_gold)))

    N_t_gold = len(all_target_spans_gold)
    N_a_gold = len(all_argument_spans_gold)
    total_pair_loss = torch.tensor(0.0, device=device)

    if N_t_gold > 0 and N_a_gold > 0:
        tgt_vecs = []
        for (b, ts, te) in all_target_spans_gold:
            vec = model._get_span_representation(sequence_output, ts, te, b)
            tgt_vecs.append(vec)
        arg_vecs = []
        for (b, as_, ae) in all_argument_spans_gold:
            vec = model._get_span_representation(sequence_output, as_, ae, b)
            arg_vecs.append(vec)

        target_mat = torch.stack(tgt_vecs, dim=0)  # [N_t_gold, H*3]
        argument_mat = torch.stack(arg_vecs, dim=0)  # [N_a_gold, H*3]

        pair_logits = model.biaffine_pairing(target_mat, argument_mat)  # [N_t_gold, N_a_gold]

        gold_pair_labels = torch.zeros_like(pair_logits, device=device)
        for i_idx, (b_t, ts, te) in enumerate(all_target_spans_gold):
            for j_idx, (b_a, as_, ae) in enumerate(all_argument_spans_gold):
                if b_t == b_a: # Ensure same batch item
                    for quad in quads_labels_batch[b_t]: # Check against original quads for this item
                        if (quad['t_start_token'].item() == ts and quad['t_end_token'].item() == te and
                            quad['a_start_token'].item() == as_ and quad['a_end_token'].item() == ae):
                            gold_pair_labels[i_idx, j_idx] = 1.0
                            break
        total_pair_loss = pair_loss_fct(pair_logits, gold_pair_labels)


    # ---- 4. Classification Loss (Group + Hate) ----
    # This part also uses gold spans from quads_labels_batch
    all_span_reprs_for_cls = []
    all_group_labels = []
    all_hateful_labels = []
    for b, quads_list in enumerate(quads_labels_batch):
        for quad in quads_list:
            ts, te = quad['t_start_token'].item(), quad['t_end_token'].item()
            as_, ae = quad['a_start_token'].item(), quad['a_end_token'].item()

            # Ensure spans are valid before getting representation
            if ts != -1 and te != -1 and as_ != -1 and ae != -1 and \
               0 <= ts <= te < seq_len and 0 <= as_ <= ae < seq_len :
                tgt_vec = model._get_span_representation(sequence_output, ts, te, b)
                arg_vec = model._get_span_representation(sequence_output, as_, ae, b)
                cls_vec = cls_output[b]
                all_span_reprs_for_cls.append(torch.cat([tgt_vec, arg_vec, cls_vec], dim=-1))
                all_group_labels.append(torch.tensor(quad['group_vector'], device=device, dtype=torch.float))
                all_hateful_labels.append(torch.tensor([quad['hateful_flag']], device=device, dtype=torch.float))

    total_group_loss = torch.tensor(0.0, device=device)
    total_hateful_loss = torch.tensor(0.0, device=device)
    if all_span_reprs_for_cls:
        span_cat = torch.stack(all_span_reprs_for_cls, dim=0)
        group_logits = model.group_classifier(span_cat)
        hateful_logits = model.hateful_classifier(span_cat)

        stacked_group_labels = torch.stack(all_group_labels, dim=0)
        stacked_hateful_labels = torch.stack(all_hateful_labels, dim=0).view(-1)

        total_group_loss = group_loss_fct(group_logits, stacked_group_labels)
        total_hateful_loss = hateful_loss_fct(hateful_logits.view(-1), stacked_hateful_labels)

    # ---- 5. 加权合并 ----
    loss = (
            span_weight * total_span_loss +
            biaffine_weight * total_pair_loss +
            group_weight * total_group_loss +
            hateful_weight * total_hateful_loss
    )

    loss_components = {
        'span_loss': total_span_loss.item(), # New BIO span loss
        'biaffine_loss': total_pair_loss.item(),
        'group_loss': total_group_loss.item(),
        'hateful_loss': total_hateful_loss.item()
    }
    return loss, loss_components
