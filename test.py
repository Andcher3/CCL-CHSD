import json
from typing import List, Dict, Any

def load_data(path: str) -> List[Dict[str, Any]]:
    # read json file to str
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(type(data), type(data[0]))
    print(data[0].keys())

    for idx in range(len(data)):
        if data[idx]['id'] == 5219:
            for key in data[idx].keys():
                print(data[idx][key])

    return data

if __name__ == '__main__':
    path = 'data/train.json'
    load_data(path)

