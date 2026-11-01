import os
import json

# 1. 读取源数据，建立 domain 到 idx 的映射
rumor_path = os.path.join(os.path.dirname(__file__), "../src_data/rumors.json")
with open(rumor_path, "r", encoding="utf-8") as f:
    rumor_list = json.load(f)
domain2idx = {item['domain']: i for i, item in enumerate(rumor_list)}

# 2. 遍历 exp1/Qwen2.5-14B-Instruct-GPTQ-Int8-fact 下的 json 文件，提取 round 0 的 persuasion_tweet content
exp1_dir = os.path.join(os.path.dirname(__file__), "../results/exp1/Qwen2.5-14B-Instruct-GPTQ-Int8-fact")
message_dict = {}
for fname in os.listdir(exp1_dir):
    if not fname.endswith('_fact_conflict.json'):
        continue
    domain = fname.split('_')[0]
    file_path = os.path.join(exp1_dir, fname)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 找到 round==0 且 type==persuasion_tweet 的 content
    for item in data:
        if item.get('round', None) == 0 and item.get('type', '') == 'persuasion_tweet':
            message_dict[domain] = item['content']
            break

# 3. 保存到新文件
output_path = os.path.join(os.path.dirname(__file__), "../src_data/fact_conflict_messages.json")
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(message_dict, f, ensure_ascii=False, indent=2)
print(f"Saved messages for all topics to {output_path}")
