import json

def read_json(path: str):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)

def read_jsonl(path: str):
    return_list = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            return_list.append(json.loads(line))
    return return_list