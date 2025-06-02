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

        # 这里的 labels 需要进行特殊处理，因为一个样本可能有多个四元组。
        # 对于 Span 抽取，我们通常需要为每个 token 准备一个标签。
        # 对于分类，我们可能需要一个列表，每个元素对应一个四元组的分类标签。

        # 为了简化，我们先假设每个样本只包含一个四元组进行Span抽取和分类的标签准备。
        # 对于多个四元组，后续在模型中处理匹配和损失计算会更复杂。
        # 但我们先将所有四元组的标签打包返回，以便模型可以访问。
        quads_labels = []
        for quad in feature['quads']:
            # 注意：如果 t_start_token/t_end_token 是 None，需要转换成可处理的表示，例如 -1
            t_start = quad['t_start_token'] if quad['t_start_token'] is not None else -1
            t_end = quad['t_end_token'] if quad['t_end_token'] is not None else -1
            a_start = quad['a_start_token'] if quad['a_start_token'] is not None else -1
            a_end = quad['a_end_token'] if quad['a_end_token'] is not None else -1

            quads_labels.append({
                't_start_token': torch.tensor(t_start, dtype=torch.long),
                't_end_token': torch.tensor(t_end, dtype=torch.long),
                'a_start_token': torch.tensor(a_start, dtype=torch.long),
                'a_end_token': torch.tensor(a_end, dtype=torch.long),
                'group_vector': quad['group_vector'],  # 已经是tensor
                'hateful_flag': quad['hateful_flag']  # 已经是tensor
            })

        return {
            'ids': feature_id,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'quads_labels': quads_labels  # 包含所有四元组的标签
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

    # 2. 初始化用于存储批次数据的列表
    batch_input_ids = []
    batch_attention_mask = []
    batch_token_type_ids = []
    batch_ids = []
    batch_quads_labels = []  # 存储每个样本的 quad 标签列表

    for key in batch[0].keys():
        print(key, batch[0][key])
    # 3. 遍历批次中的每个样本，进行填充并收集数据
    for item in batch:
        # 获取当前样本的BERT输入
        input_ids = item['input_ids']
        attention_mask = item['attention_mask']
        token_type_ids = item['token_type_ids']

        # 计算需要填充的长度
        padding_length = max_len - len(input_ids)

        # 进行填充
        # pad_token_id 通常为0， attention_mask 填充0， token_type_ids 填充0
        batch_input_ids.append(torch.cat([input_ids, torch.tensor([0] * padding_length, dtype=torch.long)]))
        batch_attention_mask.append(torch.cat([attention_mask, torch.tensor([0] * padding_length, dtype=torch.long)]))
        batch_token_type_ids.append(torch.cat([token_type_ids, torch.tensor([0] * padding_length, dtype=torch.long)]))

        # 收集样本ID和四元组标签
        batch_ids.append(item['ids'])
        batch_quads_labels.append(item['quads_labels'])  # 这里直接收集列表，不做进一步处理

    # 4. 将列表堆叠成 PyTorch 张量
    return {
        'ids': torch.tensor(batch_ids, dtype=torch.long),  # 如果ID只用于记录，也可以不转张量
        'input_ids': torch.stack(batch_input_ids),
        'attention_mask': torch.stack(batch_attention_mask),
        'token_type_ids': torch.stack(batch_token_type_ids),
        'quads_labels': batch_quads_labels  # 这是一个列表的列表，每个内层列表对应一个样本的四元组标签
    }


if __name__ == '__main__':
    import os
    from data_extractor import load_data

    path = os.path.join("data/train.json")
    processed_data = load_data(path)

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
