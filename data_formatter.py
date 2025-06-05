# from transformers import BertTokenizer # <-- 删除或注释掉这一行
from transformers import BertTokenizerFast  # <-- 修改为导入 BertTokenizerFast
import torch
from typing import List, Dict, Any
from Hype import *


# 1. 加载BERT分词器
tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')  # 或者 'hfl/chinese-bert-wwm-ext'


def convert_samples_to_features(processed_data: List[Dict[str, Any]], tokenizer: BertTokenizerFast = tokenizer,
                                max_seq_length: int = MAX_SEQ_LENGTH):  # 注意类型提示也相应修改
    features = []
    for sample in processed_data:
        content = sample['content']
        quads = sample['quads']

        encoded_content = tokenizer.encode_plus(
            content,
            max_length=max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',  # 返回PyTorch张量
            return_offsets_mapping=True,  # 字符偏移量，用于映射
            return_token_type_ids=True,  # 返回token_type_ids
            return_attention_mask=True  # 返回attention_mask
        )

        input_ids = encoded_content['input_ids'].squeeze()
        attention_mask = encoded_content['attention_mask'].squeeze()
        token_type_ids = encoded_content['token_type_ids'].squeeze()
        # offset_mapping 是 (start_char_idx, end_char_idx) for each token (end_char_idx is exclusive)
        offset_mapping = encoded_content['offset_mapping'].squeeze().tolist()

        # 为每个四元组转换Span索引
        tokenized_quads = []
        for quad in quads:
            t_start_char, t_end_char = quad['t_start'], quad['t_end']  # char_end 是包含的
            a_start_char, a_end_char = quad['a_start'], quad['a_end']  # char_end 是包含的

            # 使用 BertTokenizerFast 提供的 char_to_token 方法更鲁棒地进行映射
            t_start_token, t_end_token = None, None
            if t_start_char is not None and t_end_char is not None:
                # `char_to_token` 映射字符索引到其对应的token索引
                # 它会考虑 [CLS], [SEP] 等特殊token的偏移
                # 对于起始位置，直接查找其对应的token索引
                t_start_token = encoded_content.char_to_token(t_start_char)
                # 对于结束位置，我们需要找到覆盖到 char_end 的那个token的索引
                # 简单来说，就是从 t_start_token 开始，找到第一个其字符结束位置（exclusive）
                # 超过 (t_end_char + 1) 的 token的前一个token，或者就是包含 t_end_char 的 token

                # BertTokenizerFast.char_to_token(char_index, sequence_index=0)
                # char_index是字符在原始字符串中的索引
                # sequence_index=0表示是第一个句子（对于单句输入）

                # 找到span的起始token
                start_token_idx = encoded_content.char_to_token(t_start_char, sequence_index=0)
                # 找到span的结束token。这里的char_end是闭区间，所以要找包含这个字符的token。
                # 如果是子词分词，一个词可能被分成多个token，我们要的是包含原始词汇最后一个字符的那个token。
                end_token_idx = encoded_content.char_to_token(t_end_char, sequence_index=0)

                # 确保start_token_idx和end_token_idx都有效，并且end_token_idx不小于start_token_idx
                if start_token_idx is not None and end_token_idx is not None and end_token_idx >= start_token_idx:
                    t_start_token = start_token_idx
                    t_end_token = end_token_idx
                else:  # 如果映射失败，可能因为截断或Span在特殊token区
                    t_start_token, t_end_token = None, None  # 重置为None

            a_start_token, a_end_token = None, None
            if a_start_char is not None and a_end_char is not None:
                start_token_idx = encoded_content.char_to_token(a_start_char, sequence_index=0)
                end_token_idx = encoded_content.char_to_token(a_end_char, sequence_index=0)

                if start_token_idx is not None and end_token_idx is not None and end_token_idx >= start_token_idx:
                    a_start_token = start_token_idx
                    a_end_token = end_token_idx
                else:
                    a_start_token, a_end_token = None, None

            # 存储转换后的四元组信息
            tokenized_quads.append({
                'target_text': quad['target_text'],
                't_start_token': t_start_token,
                't_end_token': t_end_token,
                'argument_text': quad['argument_text'],
                'a_start_token': a_start_token,
                'a_end_token': a_end_token,
                'group_vector': torch.tensor(quad['group_vector'], dtype=torch.float),
                'hateful_flag': torch.tensor(quad['hateful_flag'], dtype=torch.long)
            })

        features.append({
            'id': sample['id'],
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'quads': tokenized_quads
        })
    return features


if __name__ == '__main__':
    import os
    from data_extractor import load_data

    path = os.path.join("data/train.json")
    processed_data, _ = load_data(path)

    # 显示示例（前2条）
    for item in processed_data:
        if item['id'] == 7299:
            print(f"ID: {item['id']}")
            print(f"Content: {item['content']}")
            for q in item['quads']:
                print("  Target:", q['target_text'], "Span:", (q['t_start'], q['t_end']))
                print("  Argument:", q['argument_text'], "Span:", (q['a_start'], q['a_end']))
                print("  Group vector:", q['group_vector'], "Hateful:", q['hateful_flag'])
            print("---")

    features = convert_samples_to_features(processed_data, tokenizer, MAX_SEQ_LENGTH)

    for item in features:
        if item['id'] == 7299:
            print(f"ID: {item['id']}")
            print(f"input_ids: {item['input_ids']}")
            for q in item['quads']:
                print("  Target:", q['target_text'], "Span:", (q['t_start_token'], q['t_end_token']))
                print("  Argument:", q['argument_text'], "Span:", (q['a_start_token'], q['a_end_token']))
                print("  Group vector:", q['group_vector'], "Hateful:", q['hateful_flag'])
            print("---")
