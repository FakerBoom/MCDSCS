import json
import random

# 文件路径
json_file_path = 'output_file.json'
jsonl_file_path = 'output_file.jsonl'
train_file_path = 'train.jsonl'
dev_file_path = 'dev.jsonl'
test_file_path = 'test.jsonl'

# 读取JSON文件
with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 将JSON文件转为JSONL文件
with open(jsonl_file_path, 'w', encoding='utf-8') as f:
    for entry in data:
        json.dump(entry, f, ensure_ascii=False)
        f.write('\n')

# 按比例随机划分数据
random.shuffle(data)
total = len(data)
train_end = int(total * 0.8)
dev_end = int(total * 0.9)

train_data = data[:train_end]
dev_data = data[train_end:dev_end]
test_data = data[dev_end:]

# 将划分后的数据写入JSONL文件
with open(train_file_path, 'w', encoding='utf-8') as f:
    for entry in train_data:
        json.dump(entry, f, ensure_ascii=False)
        f.write('\n')

with open(dev_file_path, 'w', encoding='utf-8') as f:
    for entry in dev_data:
        json.dump(entry, f, ensure_ascii=False)
        f.write('\n')

with open(test_file_path, 'w', encoding='utf-8') as f:
    for entry in test_data:
        json.dump(entry, f, ensure_ascii=False)
        f.write('\n')

print(f"数据已成功保存到 {train_file_path}, {dev_file_path}, {test_file_path}")
