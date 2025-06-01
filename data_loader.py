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
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'quads_labels': quads_labels  # 包含所有四元组的标签
        }


if __name__ == '__main__':
    import os
    from data_extractor import load_data

    path = os.path.join("data/train.json")
    processed_data = load_data(path)

    from data_formatter import convert_samples_to_features

    features = convert_samples_to_features(processed_data)
    # 后续可以这样使用：
    dataset = CHSDDataset(features)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=...)
    # 注意：这里需要一个自定义的 collate_fn 来处理 batch 中每个样本 quad_labels 数量不一致的问题
