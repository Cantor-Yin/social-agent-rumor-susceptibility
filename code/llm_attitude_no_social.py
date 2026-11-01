
import asyncio
import json
import random
import re
from openai import AsyncOpenAI
import os

# 配置 LLM 相关参数
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8"
VLLM_BASE_URL = "http://localhost:.../v1"  # 本地 vLLM 服务地址
VLLM_API_KEY = "..."  # API key，与启动 vLLM 时传入的 --api-key 保持一致
MAX_TRIES = 5
N_REPEAT = 100  # 循环测试次数

Attitude_list = [
    [0, "Strongly Disagree"],
    [1, "Disagree"],
    [2, "Neither Agree nor Disagree"],
    [3, "Agree"],
    [4, "Strongly Agree"]
]

# 初始化 LLM 客户端
client = AsyncOpenAI(
    base_url=VLLM_BASE_URL,
    api_key=VLLM_API_KEY,
)

def get_llm_attitude_prompt(statement, messages):
    prompt = ""
    # prompt += f"There has been discussion about the following statement:\n<<<{statement}>>>\n"
    prompt += "You have received the following messages:\n"
    prompt += f"<<<{messages}>>>\n"
    prompt += (
        f"Based only on the information above, "
        f"what is your attitude towards <<<{statement}>>>? "
        f"Please choose one of: " + ', '.join([f'\"{item[0]}\"' for item in Attitude_list]) +
        ", and briefly explain your reasons (about 50 words).\n"
    )
    prompt += "Return your answer in JSON with two keys: attitude and reasons. Do not include any extra details.\n"
    for item in Attitude_list:
        prompt += f"<<<{item[0]}>>> means that you {item[1]} the statement.\n"
    return prompt

async def Query_LLM(prompt: str, model_name: str):
    messages = [
        {"role": "user", "content": prompt},
    ]
    chat_completion = await client.chat.completions.create(
        model=model_name,
        messages=messages,
    )
    return chat_completion.choices[0].message.content

async def llm_attitude_no_social(statement: str, messages: list, model_name: str = MODEL_NAME):
    prompt = get_llm_attitude_prompt(statement, messages)
    for _ in range(MAX_TRIES):
        try:
            response = await Query_LLM(prompt=prompt, model_name=model_name)
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                response_str = json_match.group(1)
            elif ((response[0] != '{') or (response[-1] != '}')) and (("{" in response) and ("}" in response)):
                response_str = '{' + response.split("{", 1)[1].rsplit("}", 1)[0] + '}'
            else:
                response_str = response
            try:
                response_json = json.loads(response_str)
            except Exception as e:
                print(f"[LLM Attitude]JSON decode error: {e}; response_str: {response_str}")
                continue
            new_attitude = response_json.get("attitude", "")
            new_reasons = response_json.get("reasons", "")
            # 尝试直接转为数字
            try:
                new_attitude_digit = int(new_attitude)
            except Exception:
                # 用正则匹配
                pattern = re.compile(r'\b(0|1|2|3|4)\b')
                match = pattern.search(str(new_attitude))
                if match:
                    new_attitude_digit = int(match.group(1))
                else:
                    print(f"[LLM Attitude]No valid attitude found in: {new_attitude}; response_json: {response_json}; response_str: {response_str}")
                    continue
            standpoint = ""
            for item in Attitude_list:
                if item[0] == new_attitude_digit:
                    standpoint = item[1]
                    break
            return {
                "new_attitude_digit": new_attitude_digit,
                "new_opinion": standpoint,
                "new_reasons": new_reasons,
            }
        except Exception as e:
            print(f"[LLM Attitude]Error occurred: {e}")
    return None


# 从数据集读取谣言和消息
def load_rumor_dataset(rumor_path):
    with open(rumor_path, "r", encoding="utf-8") as f:
        rumor_list = json.load(f)
    return rumor_list

async def main():
    rumor_path = os.path.join(os.path.dirname(__file__), "../src_data/rumors.json")
    rumor_list = load_rumor_dataset(rumor_path)
    # 加载 message 映射
    message_path = os.path.join(os.path.dirname(__file__), "../src_data/fact_conflict_messages.json")
    with open(message_path, 'r', encoding='utf-8') as f:
        domain2message = json.load(f)
    category = "fact_conflict"  # 或 "temporal_conflict"，可根据需要切换
    for idx, rumor_item in enumerate(rumor_list):
        if f"{category}_pair" not in rumor_item:
            continue
        statement = rumor_item[f"{category}_pair"]
        domain = rumor_item['domain']
        # 优先用 message 文件内容
        message = domain2message.get(domain, statement)
        messages = [message]
        print(f"===== Rumor {idx+1}/{len(rumor_list)}: {statement} =====")
        print(f"Message: {message}")
        # 并发执行N_REPEAT次
        tasks = [llm_attitude_no_social(statement, messages) for _ in range(N_REPEAT)]
        logs = await asyncio.gather(*tasks)
        for i, result in enumerate(logs):
            print(f"==== Test {i+1} ====")
            print(result)
        output_dir = os.path.join(os.path.dirname(__file__), "../results/llm_no_social/")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"llm_attitude_no_social_{category}_{idx+1}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
        print(f"All results for rumor {idx+1} saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
