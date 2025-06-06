from torch.utils.data import Dataset, DataLoader
import torch
from typing import List, Dict, Any
from Hype import *


class CHSDDataset(Dataset):
    def __init__(self, features: List[Dict[str, Any]]):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]

        feature_id = feature['id']
        input_ids = feature['input_ids']
        attention_mask = feature['attention_mask']
        token_type_ids = feature['token_type_ids']

        original_content = feature.get('content', '')  # 从 features 获取原始 content

        quads_labels = []
        for quad in feature['quads']:
            t_start = quad['t_start_token'] if quad['t_start_token'] is not None else -1
            t_end = quad['t_end_token'] if quad['t_end_token'] is not None else -1
            a_start = quad['a_start_token'] if quad['a_start_token'] is not None else -1
            a_end = quad['a_end_token'] if quad['a_end_token'] is not None else -1

            quads_labels.append({
                't_start_token': torch.tensor(t_start, dtype=torch.long),
                't_end_token': torch.tensor(t_end, dtype=torch.long),
                'a_start_token': torch.tensor(a_start, dtype=torch.long),
                'a_end_token': torch.tensor(a_end, dtype=torch.long),
                'group_vector': torch.tensor(quad['group_vector'], dtype=torch.float),  # 确保是 tensor
                'hateful_flag': torch.tensor(quad['hateful_flag'], dtype=torch.long),  # 确保是 tensor
                'target_text_raw': quad['target_text'],  # 原始 char-level 的 target_text
                'argument_text_raw': quad['argument_text']  # 原始 char-level 的 argument_text
            })

        return {
            'ids': feature_id,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'original_content': original_content,  # 返回原始 content 字符串
            'quads_labels': quads_labels
        }


# 自定义的 collate_fn 来处理 batch 中每个样本 quad_labels 数量不一致的问题
def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collates a list of processed features into a batch for training.

    Args:
        batch: A list of dictionaries, where each dictionary represents a sample
               from the output of convert_samples_to_features.
               Each sample dict should contain:
               - 'id': int
               - 'input_ids': torch.Tensor
               - 'attention_mask': torch.Tensor
               - 'token_type_ids': torch.Tensor
               - 'quads': List[Dict] (list of token-level quads for this sample)

    Returns:
        A dictionary containing padded tensors for BERT inputs and
        a list of lists for the quad labels.
    """
    # 1. 找到当前批次中最长的序列长度
    max_len = max(len(item['input_ids']) for item in batch)

    batch_input_ids = []
    batch_attention_mask = []
    batch_token_type_ids = []
    batch_ids = []
    batch_quads_labels = []
    batch_original_content = []  # 新增

    for item in batch:
        input_ids = item['input_ids']
        attention_mask = item['attention_mask']
        token_type_ids = item['token_type_ids']

        padding_length = max_len - len(input_ids)

        batch_input_ids.append(torch.cat([input_ids, torch.tensor([0] * padding_length, dtype=torch.long)]))
        batch_attention_mask.append(torch.cat([attention_mask, torch.tensor([0] * padding_length, dtype=torch.long)]))
        batch_token_type_ids.append(torch.cat([token_type_ids, torch.tensor([0] * padding_length, dtype=torch.long)]))

        batch_ids.append(item['ids'])
        batch_quads_labels.append(item['quads_labels'])
        batch_original_content.append(item['original_content'])  # 新增

    return {
        'ids': torch.tensor(batch_ids, dtype=torch.long),
        'input_ids': torch.stack(batch_input_ids),
        'attention_mask': torch.stack(batch_attention_mask),
        'token_type_ids': torch.stack(batch_token_type_ids),
        'original_content': batch_original_content,  # 返回字符串列表
        'quads_labels': batch_quads_labels
    }


if __name__ == '__main__':
    import os
    from data_extractor import load_data

    path = os.path.join("data/train.json")
    processed_data, _ = load_data(path)

    from data_formatter import convert_samples_to_features

    features = convert_samples_to_features(processed_data)

    dataset = CHSDDataset(features)
    # dataset: class which features is [{ids, input_ids, attention_mask(where is str), token_type_ids(),
    # quads:[
    #       {target_text, target_start/end_token, argument_text,
    #       argument_start/end_token, group_multi_hot_encode, hateful}
    #       ]
    # }]
    # print(dataset.features)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    print("=== 检查 DataLoader 中第一个 batch 的内容 ===")
    # batch结构同上
    for batch in dataloader:
        print("Batch keys:", batch.keys())
        print("Batch input_ids shape:", batch['input_ids'].shape)
        print("Batch attention_mask shape:", batch['attention_mask'].shape)
        print("Batch token_type_ids shape:", batch['token_type_ids'].shape)
        print("Batch quads_labels 示例：")

        for i, sample_quads in enumerate(batch['quads_labels']):
            print(f"  样本 {i} 含 {len(sample_quads)} 个四元组")
            for quad in sample_quads:
                if quad['t_start_token'].item() == -1:
                    print(batch['ids'][i])
                print("    T-span:", (quad['t_start_token'].item(), quad['t_end_token'].item()),
                      "A-span:", (quad['a_start_token'].item(), quad['a_end_token'].item()),
                      "Group:", quad['group_vector'].tolist(),
                      "Hate:", quad['hateful_flag'].item())
        break  # 只看第一个 batch
