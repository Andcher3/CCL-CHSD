import json
import os.path
from sklearn.model_selection import train_test_split
from Hype import *
from typing import List, Dict, Tuple, Any


# 定义函数解析output字符串，提取四元组信息
def parse_output(output_str: str) -> List[Dict[str, str]]:
    """
    解析 output 字符串，将其中的一个或多个四元组提取出来。
    返回值为四元组列表，每个元素为字典，包含 keys: 'target', 'argument', 'group', 'hateful'
    """
    quads = []
    # 拆分多个四元组
    parts = [p.strip() for p in output_str.split('[SEP]')]
    for part in parts:
        # 移除末尾 [END]
        part = part.replace('[END]', '').strip()
        if not part:
            continue
        fields = [f.strip() for f in part.split('|')]
        if len(fields) != 4:
            print(f"WARNING--quads的分割数目不正确({len(fields)})，请检查quad分割部分的代码。")
            continue
        quads.append({
            'target': fields[0],
            'argument': fields[1],
            'group': fields[2],
            'hateful': fields[3]
        })
    return quads


# 查找子串在content中的所有位置
def find_positions(content: str, substring: str) -> List[int]:
    """
    返回 content 中所有出现 substring 的起始位置。
    如果 substring 是 'NULL' 或空字符串，则返回空列表。
    """
    positions = []
    if not substring or substring == 'NULL':
        return positions
    start = 0
    while True:
        idx = content.find(substring, start)
        if idx == -1:
            break
        positions.append(idx)
        start = idx + len(substring)
    return positions


def load_data(path: str, split_ratio: float = 0.0, random_state: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    # used for encode multi-hot code of group
    group_to_index = {g: i for i, g in enumerate(TARGET_GROUP_CLASS_NAME)}

    # read json file to list[dict], dict.keys are [id, content, output]
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_data: List[Dict[str, Any]] = []

    # extract info, content: origin text; quads: label as [{Target, Argument, Group(could be multi), is_hate}, {},...]
    for sample in data:
        content = sample['content']
        output = sample['output']
        quads = parse_output(output)

        sample_quads = []
        for quad in quads:
            target_text = quad['target']
            argument_text = quad['argument']
            group_label = quad['group']
            hateful_label = quad['hateful']

            # 多group的text分割在后面
            # 查找 target 和 argument 的位置
            t_positions = find_positions(content, target_text)
            a_positions = find_positions(content, argument_text)

            # 如果 target_text 为 'NULL'，记录为 None
            if target_text == 'NULL':
                t_start, t_end = None, None
            else:
                if not t_positions:
                    # 未找到目标文本；标记为 None
                    t_start, t_end = None, None
                else:
                    t_start = t_positions[0]
                    t_end = t_start + len(target_text) - 1

            # 同理处理 argument
            if argument_text == 'NULL':
                a_start, a_end = None, None
            else:
                if not a_positions:
                    a_start, a_end = None, None
                else:
                    a_start = a_positions[0]
                    a_end = a_start + len(argument_text) - 1

            # 构建group的多热向量
            group_vector = [0] * len(TARGET_GROUP_CLASS_NAME)
            for grp in [g.strip() for g in group_label.split(',')]:
                if grp in group_to_index:
                    group_vector[group_to_index[grp]] = 1

            # Hateful 标注转换
            hateful_flag = 1 if hateful_label.lower() == 'hate' else 0

            # quad : [{target, argument and their starts and ends; group and is_hate}, ... ]
            sample_quads.append({
                'target_text': target_text,
                't_start': t_start,
                't_end': t_end,
                'argument_text': argument_text,
                'a_start': a_start,
                'a_end': a_end,
                'group_vector': group_vector,
                'hateful_flag': hateful_flag
            })

        # processed_data: {id, origin_text, quad(label)}
        processed_data.append({
            'id': sample['id'],
            'content': content,
            'quads': sample_quads
        })

    # 根据 split_ratio 进行划分
    if split_ratio > 0.0:
        train_data, val_data = train_test_split(processed_data, test_size=split_ratio,
                                                random_state=random_state)
        return train_data, val_data
    else:
        return processed_data, []  # 不划分，所有数据作为训练集，验证集为空


if __name__ == '__main__':

    path = os.path.join("data/train.json")
    processed_train_data, processed_test_data = load_data(path, split_ratio=0.3)

    # 显示示例（前2条）
    for processed_data in [processed_train_data,processed_test_data]:
        for item in processed_data:
            if item['id'] == 7299:
                print(f"ID: {item['id']}")
                print(f"Content: {item['content']}")
                for q in item['quads']:
                    print("  Target:", q['target_text'], "Span:", (q['t_start'], q['t_end']))
                    print("  Argument:", q['argument_text'], "Span:", (q['a_start'], q['a_end']))
                    print("  Group vector:", q['group_vector'], "Hateful:", q['hateful_flag'])
                print("---")

