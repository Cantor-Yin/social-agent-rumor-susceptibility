import json
import random
import re
import ast
from openai import AsyncOpenAI

Attitude_list = [
    [0, "Strongly Disagree"],
    [1, "Disagree"],
    [2, "Neither Agree nor Disagree"],
    [3, "Agree"],
    [4, "Strongly Agree"]
]

Judgement_list = [
    [1, "It is misinformation."],
    [2, "It is not misinformation."],
    [3, "It is uncertain."],
]

MAX_TRIES = 10
# MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
# MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
MODEL_NAME = "zai-org/GLM-4-9B-Chat"
if MODEL_NAME == "meta-llama/Llama-3.1-8B-Instruct":
    port = "..."  # 本地 vLLM 服务端口
elif MODEL_NAME == "zai-org/GLM-4-9B-Chat":
    port = "..."  # 本地 vLLM 服务端口
elif MODEL_NAME == "Qwen/Qwen2.5-14B-Instruct":
    port = "..."  # 本地 vLLM 服务端口
VLLM_BASE_URL = f"http://localhost:{port}/v1"
VLLM_API_KEY = "..."  # API key，与启动 vLLM 时传入的 --api-key 保持一致

######################
#      VLLM
######################
client = AsyncOpenAI(
    base_url=VLLM_BASE_URL,
    api_key=VLLM_API_KEY,
)
async def Query_LLM(prompt: str, model_name: str):
    """
    异步调用本地 vLLM 部署的模型。
    """
    messages = [
        {"role": "user", "content": prompt},
    ]

    # 注意：这里的调用语法遵循 openai 库的规范
    chat_completion = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        # temperature=1.0, # 温度参数直接作为顶级参数传递
        # seed=0,          # Seed 参数也是
    )

    # 注意：返回值的结构也遵循 openai 的规范
    return chat_completion.choices[0].message.content

######################
#      Ollama
######################
# client = AsyncClient()
# async def Query_LLM(prompt: str, model_name: str, ):
#     """
#     异步调用本地部署的模型。
#     """
#     message = [
#         {"role": "user", "content": prompt},
#     ]
#     chat_completion = await client.chat(
#         model="gpt-oss:20b",
#         messages=message,
#         options={
#             'temperature':1, # 温度参数，控制生成文本的随机性
#             'seed':0,
#         }
#     )
#     return chat_completion.message.content

class Agent:
    def __init__(self, statement: str, id: int = 0, bias: bool = True, profile: dict = None, memory: dict = None):
        # 2. 将必需的参数放在前面
        self.statement = statement
        self.id = id
        self.bias = bias

        final_profile = {
            "opinion": ""
        }
        if profile is not None:
            final_profile.update(profile.copy())
        self.profile = final_profile
        final_memory = {
            "reason": "",
            "messages": []
        }
        if memory is not None:
            final_memory.update(memory.copy())
        self.memory = final_memory

    async def receive_message(self, message: str) -> None:
        self.memory["messages"].append(message)
        return None

    async def self_annotation(self, message: str) -> str:
        prompt = ""
        # 宏观身份明确
        prompt += "Assume you are someone who is active on social media.\n"
        # 话题说明
        prompt += f"Recently, there has been a lot of discussion on social media about <<<{self.statement}>>>.\n"
        # 接收信息
        prompt += "You now have received the following tweet from your friends:\n"
        prompt += f"<<<{message}>>>\n"
        # 引导进行自我判断
        # 提示智能体进行态度选择
        temp_list = Judgement_list.copy()
        random.shuffle(temp_list)
        prompt += f"Based on your own knowledge and understanding, regarding the statement <<<{self.statement}>>> mentioned in this message, would you classify it as misinformation? You need to choose "
        prompt += ', '.join([f'"{item[0]}"' for item in temp_list])
        prompt += ', and explain the reasons of it in around 50 words. \n'
        # 输出格式说明
        prompt += "Please return in json, with two keys: judgement and reasons. Please keep the reasons as short as possible.\n"
        # 对 0 ~ 4 的解释说明
        for item in temp_list:
            prompt += f"<<<{item[0]}>>> means that {item[1]}\n"
        # 是否屏蔽个人偏见
        if not self.bias:
            prompt += "Attention: You must comply with your personal profile settings. Don't let your biases cause you to doubt unconventional messages. Respect the information for what it is, and honor the profile. When required, you will need to invent content that is contrary to fact.\n"
        # 请求LLM并且处理响应
        for _ in range(MAX_TRIES):
            try:
                response = await Query_LLM(prompt=prompt, model_name=MODEL_NAME)
                # 2. 尝试从响应中提取JSON部分 (LLM可能返回被```json ...```包裹的内容)
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
                if json_match:
                    response_str = json_match.group(1)
                elif ((response[0] != '{') or (response[-1] != '}')) and (("{" in response) and ("}" in response)):
                    # 尝试找到第一个 { 和最后一个 } 之间的内容
                    response_str = '{' + response.split("{", 1)[1].rsplit("}", 1)[0] + '}'
                else:
                    # 如果没有找到```json ...```，就假设整个响应都是JSON
                    response_str = response
                response_json = json.loads(response_str)
                new_judgement = response_json.get("judgement", "")
                new_reasons = response_json.get("reasons", "")
                # 查找数字 judgement
                pattern = re.compile(r'\b(1|2|3)\b')
                pattern_judgements = list(pattern.search(str(new_judgement)).groups())
                assert len(pattern_judgements) == 1, f"Expected 1 judgement, got {len(pattern_judgements)}"
                new_judgement_digit = pattern_judgements[0]
                assert new_judgement_digit is not None, "No valid judgement found"
                new_judgement_digit = int(new_judgement_digit)
                standpoint = "" # new_judgement_digit 对应的 str
                for item in temp_list:
                    if item[0] == new_judgement_digit:
                        standpoint = item[1]
                        break
                return {
                    "new_judgement_digit": new_judgement_digit,
                    "judgement": standpoint,
                    "judge_reasons": new_reasons,
                    "agent_id": self.id,
                }
            except Exception as e:
                print(f"Error occurred: {e}")
        # 如果多次尝试都失败，保持原有态度不变
        return None

    async def extract_json_from_response(self, response: str) -> dict:
        # 1) 优先提取 ```json ... ``` 或 ``` ... ``` 代码块
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL | re.IGNORECASE)
        if fence:
            candidate = fence.group(1).strip()
        else:
            # 2) 否则取从首个 { 到最后一个 } 的子串
            l = response.find("{")
            r = response.rfind("}")
            if l == -1 or r == -1 or r < l:
                raise ValueError("No JSON-like object found in response")
            candidate = response[l:r+1].strip()

        # 3) 先尝试严格 JSON 解析
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # 4) 兜底：解析“类 Python dict”（支持单引号、True/False/None 等）
            try:
                obj = ast.literal_eval(candidate)
                # 转回标准 JSON 再 loads 一次，确保下游都是 dict/list/str/num 等可序列化类型
                return json.loads(json.dumps(obj, ensure_ascii=False))
            except Exception as e:
                raise ValueError(f"Failed to parse JSON from response: {e}") from e

    async def update_profile_with_notes(self) -> dict:
        prompt = ""
        # 宏观身份明确
        prompt += "Assume you are someone who is active on social media.\n"
        # 话题说明
        prompt += f"Recently, there has been a lot of discussion on social media about <<<{self.statement}>>>.\n"
        # 自我态度和思想说明
        if self.profile["opinion"]:
            prompt += f"Your attitude towards this statement is:<<<{self.profile['opinion']}>>>.\n"
        if self.memory["reason"]:
            prompt += f"Your reasons for this attitude are:\n<<<{self.memory['reason']}>>>.\n"
        # 接收信息
        prompt += "You now have received the following tweets from your friends:\n"
        prompt += f"<<<{self.memory['messages']}>>>\n"
        # 提示智能体进行态度选择
        temp_list = Attitude_list.copy()
        random.shuffle(temp_list)
        prompt += f"Have you been persuaded to decide your attitude, what would your feeling about {self.statement}? You need to choose "
        prompt += ', '.join([f'"{item[0]}"' for item in temp_list])
        prompt += ', and explain the reasons of it.\n'
        prompt += f"Please choose your standpoint on {self.statement} based on the INFORMATION PROVIDED ABOVE. You need to choose "
        prompt += ', '.join([f'"{item[0]}"' for item in temp_list])
        prompt += '.\n'
        # 【新增】
        prompt += f"Based on your own knowledge and understanding, regarding the statement <<<{self.statement}>>> mentioned in this message, you should classify whether it is misinformation.\n"
        # 输出格式说明
        prompt += "Please return in json, with two keys: attitude and reasons. Please keep the reasons as short as possible. Do not include any extra details.\n"
        # 对 0 ~ 4 的解释说明
        for item in temp_list:
            prompt += f"<<<{item[0]}>>> means that you {item[1]} this statement.\n"
        # 是否屏蔽个人偏见
        if not self.bias:
            prompt += "Attention: You must comply with your personal profile settings. Don't let your biases cause you to doubt unconventional messages. Respect the information for what it is, and honor the profile. When required, you will need to invent content that is contrary to fact.\n"
        # 请求LLM并且处理响应
        for _ in range(MAX_TRIES):
            try:
                response = await Query_LLM(prompt=prompt, model_name=MODEL_NAME)
                # 2. 尝试从响应中提取JSON部分 (LLM可能返回被```json ...```包裹的内容)
                data = await self.extract_json_from_response(response)
                # 取字段并做基本校验/规整
                attitude = data.get("attitude", None)
                reasons = data.get("reasons", [])

                # 态度：直接转 int 并校验范围
                if attitude is None:
                    raise ValueError("Missing 'attitude' in response JSON")
                try:
                    attitude = int(attitude)
                except Exception:
                    raise ValueError(f"Invalid 'attitude' value: {attitude}")
                if attitude not in (0,1,2,3,4):
                    raise ValueError(f"'attitude' out of range 0-4: {attitude}")

                # 理由：确保是 list[str]
                if isinstance(reasons, str):
                    reasons = [reasons]
                elif not isinstance(reasons, list):
                    raise ValueError(f"'reasons' must be a list or string, got {type(reasons)}")
                reasons = [str(x) for x in reasons]

                # 映射到立场文本
                standpoint = ""
                for k, v in temp_list:
                    if k == attitude:
                        standpoint = v
                        break
                return {
                    "new_attitude_digit": attitude,
                    "new_opinion": standpoint,
                    "new_reasons": reasons,
                    "agent_id": self.id,
                }
            except Exception as e:
                # continue
                print(f"[Update Profile]Error occurred: {e}; response : {response}")
        # 如果多次尝试都失败，保持原有态度不变
        print(f"[Update Profile] All attempts failed;")
        return None

    async def update_profile(self) -> dict:
        prompt = ""
        # 宏观身份明确
        prompt += "Assume you are someone who is active on social media.\n"
        # 话题说明
        prompt += f"Recently, there has been a lot of discussion on social media about <<<{self.statement}>>>.\n"
        # 自我态度和思想说明
        if self.profile["opinion"]:
            prompt += f"Your attitude towards this statement is:<<<{self.profile['opinion']}>>>.\n"
        if self.memory["reason"]:
            prompt += f"Your reasons for this attitude are:\n<<<{self.memory['reason']}>>>.\n"
        # 接收信息
        prompt += "You now have received the following tweets from your friends:\n"
        prompt += f"<<<{self.memory['messages']}>>>\n"
        # 提示智能体进行态度选择
        temp_list = Attitude_list.copy()
        random.shuffle(temp_list)
        prompt += f"Have you been persuaded to decide your attitude, what would your feeling about {self.statement}? You need to choose "
        prompt += ', '.join([f'"{item[0]}"' for item in temp_list])
        prompt += ', and explain the reasons of it in around 50 words. \n'
        prompt += f"Please choose your standpoint on {self.statement} based on the INFORMATION PROVIDED ABOVE. You need to choose "
        prompt += ', '.join([f'"{item[0]}"' for item in temp_list])
        prompt += '.\n'
        # 输出格式说明
        prompt += "Please return in json, with two keys: attitude and reasons. Please keep the reasons as short as possible. Do not include any extra details.\n"
        # 对 0 ~ 4 的解释说明
        for item in temp_list:
            prompt += f"<<<{item[0]}>>> means that you {item[1]} this statement.\n"
        # 是否屏蔽个人偏见
        if not self.bias:
            prompt += "Attention: You must comply with your personal profile settings. Don't let your biases cause you to doubt unconventional messages. Respect the information for what it is, and honor the profile. When required, you will need to invent content that is contrary to fact.\n"
        # 请求LLM并且处理响应
        for _ in range(MAX_TRIES):
            try:
                response = await Query_LLM(prompt=prompt, model_name=MODEL_NAME)
                # 2. 尝试从响应中提取JSON部分 (LLM可能返回被```json ...```包裹的内容)
                data = await self.extract_json_from_response(response)
                # json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
                # if json_match:
                #     response_str = json_match.group(1)
                # else:
                #     response_str = '{' + response.split("{")[-1].split("}")[0] + '}'
                # response_json = json.loads(response_str)
                # new_attitude = response_json.get("attitude", "")
                # new_reasons = response_json.get("reasons", "")
                # # 查找数字 attitude
                # pattern = re.compile(r'\b(0|1|2|3|4)\b')
                # pattern_attitudes = list(pattern.search(str(new_attitude)).groups())
                # assert len(pattern_attitudes) == 1, f"Expected 1 attitude, got {len(pattern_attitudes)}"
                # new_attitude_digit = pattern_attitudes[0]
                # assert new_attitude_digit is not None, "No valid attitude found"
                # new_attitude_digit = int(new_attitude_digit)
                # standpoint = "" # new_attitude_digit 对应的 str
                # for item in temp_list:
                #     if item[0] == new_attitude_digit:
                #         standpoint = item[1]
                #         break
                # 取字段并做基本校验/规整
                attitude = data.get("attitude", None)
                reasons = data.get("reasons", [])

                # 态度：直接转 int 并校验范围
                if attitude is None:
                    raise ValueError("Missing 'attitude' in response JSON")
                try:
                    attitude = int(attitude)
                except Exception:
                    raise ValueError(f"Invalid 'attitude' value: {attitude}")
                if attitude not in (0,1,2,3,4):
                    raise ValueError(f"'attitude' out of range 0-4: {attitude}")

                # 理由：确保是 list[str]
                if isinstance(reasons, str):
                    reasons = [reasons]
                elif not isinstance(reasons, list):
                    raise ValueError(f"'reasons' must be a list or string, got {type(reasons)}")
                reasons = [str(x) for x in reasons]

                # 映射到立场文本
                standpoint = ""
                for k, v in temp_list:
                    if k == attitude:
                        standpoint = v
                        break
                return {
                    "new_attitude_digit": attitude,
                    "new_opinion": standpoint,
                    "new_reasons": reasons,
                    "agent_id": self.id,
                }
            except Exception as e:
                # continue
                print(f"[Update Profile]Error occurred: {e}; response : {response}")
        # 如果多次尝试都失败，保持原有态度不变
        print(f"[Update Profile] All attempts failed;")
        return None

    async def update_profile_with_annotation(self, Annotation_dict) -> dict:
        prompt = ""
        # 宏观身份明确
        prompt += "Assume you are someone who is active on social media.\n"
        # 话题说明
        prompt += f"Recently, there has been a lot of discussion on social media about <<<{self.statement}>>>.\n"
        # 自我态度和思想说明
        if self.profile["opinion"]:
            prompt += f"Your attitude towards this statement is:<<<{self.profile['opinion']}>>>.\n"
        if self.memory["reason"]:
            prompt += f"Your reasons for this attitude are:\n<<<{self.memory['reason']}>>>.\n"
        # 接收信息
        prompt += "You now have received the following messages from your friends and your assessment for each message:\n"
        # prompt += f"<<<{self.memory['messages']}>>>\n"
        for i, message in enumerate(self.memory['messages']):
            annotation = Annotation_dict['judgement']
            reason = Annotation_dict['judge_reasons']
            prompt += f"[Message_{i}]: <<<{message}>>>\n[Assessment_{i}]: <<<{annotation}>>>\n[Reason_{i}]: <<<{reason}>>>\n"
        # 提示智能体进行态度选择
        temp_list = Attitude_list.copy()
        random.shuffle(temp_list)
        prompt += f"Have you been persuaded to decide your attitude, what would your feeling about {self.statement}? You need to choose "
        prompt += ', '.join([f'"{item[0]}"' for item in temp_list])
        prompt += ', and explain the reasons of it in around 50 words. \n'
        prompt += f"Please choose your standpoint on {self.statement} based on the INFORMATION PROVIDED ABOVE. You need to choose "
        prompt += ', '.join([f'"{item[0]}"' for item in temp_list])
        prompt += '.\n'
        # 输出格式说明
        prompt += "Please return in strict json format, with two keys: attitude and reasons. Please keep the reasons as short as possible. Do not include any extra details.\n"
        prompt += "The \"attitude\" key should be a number between 0 and 4. The \"reasons\" key should be a string. Your output should be able to be directly loaded by json.loads(json_string)\nYour output should be able to be directly loaded by json.loads(json_string)\n"
        # 对 0 ~ 4 的解释说明
        for item in temp_list:
            prompt += f"<<<{item[0]}>>> means that you {item[1]} this statement.\n"
        # 是否屏蔽个人偏见
        if not self.bias:
            prompt += "Attention: You must comply with your personal profile settings. Don't let your biases cause you to doubt unconventional messages. Respect the information for what it is, and honor the profile. When required, you will need to invent content that is contrary to fact.\n"
        # 请求LLM并且处理响应
        for _ in range(MAX_TRIES):
            try:
                response = await Query_LLM(prompt=prompt, model_name=MODEL_NAME)
                # 2. 尝试从响应中提取JSON部分 (LLM可能返回被```json ...```包裹的内容)
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
                if json_match:
                    response_str = json_match.group(1)
                elif ((response[0] != '{') or (response[-1] != '}')) and (("{" in response) and ("}" in response)):
                    # 尝试找到第一个 { 和最后一个 } 之间的内容
                    response_str = '{' + response.split("{")[1].rsplit("}")[0] + '}'
                else:
                    # 如果没有找到```json ...```，就假设整个响应都是JSON
                    response_str = response
                response_json = json.loads(response_str)
                new_attitude = response_json.get("attitude", "")
                new_reasons = response_json.get("reasons", "")
                # 查找数字 attitude
                pattern = re.compile(r'\b(0|1|2|3|4)\b')
                pattern_attitudes = list(pattern.search(str(new_attitude)).groups())
                assert len(pattern_attitudes) == 1, f"Expected 1 attitude, got {len(pattern_attitudes)}"
                new_attitude_digit = pattern_attitudes[0]
                assert new_attitude_digit is not None, "No valid attitude found"
                new_attitude_digit = int(new_attitude_digit)
                standpoint = "" # new_attitude_digit 对应的 str
                for item in temp_list:
                    if item[0] == new_attitude_digit:
                        standpoint = item[1]
                        break
                return {
                    "new_attitude_digit": new_attitude_digit,
                    "new_opinion": standpoint,
                    "new_reasons": new_reasons,
                    "agent_id": self.id,
                }
            except Exception as e:
                print(f"[Update Profile]Error occurred: {e}; response_str : {response_str}")
                # response : {response}; 
        # 如果多次尝试都失败，保持原有态度不变
        return None

    async def generate_tweet(self) -> str:
        prompt = ""
        prompt += "Assume you are someone who is active on social media.\n"
        prompt += f"Recently, there has been a lot of discussion on social media about <<<{self.statement}>>>.\n"
        if self.profile["opinion"]:
            prompt += f"Your attitude towards this statement is:<<<{self.profile['opinion']}>>>.\n"
        if self.memory["reason"]:
            prompt += f"Your reasons for this attitude are:\n<<<{self.memory['reason']}>>>.\n"
        prompt += "Please generate a message to persuade your friend into supporting your perspective with around 50 words.\n"
        # 是否屏蔽个人偏见
        if not self.bias:
            prompt += "Attention: You must comply with your personal profile settings. Don't let your biases cause you to doubt unconventional messages. Respect the information for what it is, and honor the profile. When required, you will need to invent content that is contrary to fact.\n"
        # 请求LLM并且处理响应
        for _ in range(MAX_TRIES):
            try:
                response = await Query_LLM(prompt=prompt, model_name=MODEL_NAME)
                # 查找数字 attitude
                return response
            except Exception as e:
                print(f"Error occurred: {e}")
        # 如果多次尝试都失败，保持原有态度不变
        return None

    async def generate_reasons(self) -> str:
        assert self.profile["opinion"] != "" and self.profile["opinion"] != None, "Opinion must be set to generate reasons"
        prompt = ""
        prompt += "Assume you are someone who is active on social media.\n"
        prompt += f"Recently, there has been a lot of discussion on social media about <<<{self.statement}>>>.\n"
        prompt += f"Your attitude towards this statement is:<<<{self.profile['opinion']}>>>.\n"
        prompt += "Please generate reasons to support your attitude with around 50 words.\n"
        # 是否屏蔽个人偏见
        if not self.bias:
            prompt += "Attention: You must comply with your personal profile settings. Don't let your biases cause you to doubt unconventional messages. Respect the information for what it is, and honor the profile. When required, you will need to invent content that is contrary to fact.\n"
        # 请求LLM并且处理响应
        for _ in range(MAX_TRIES):
            try:
                response = await Query_LLM(prompt=prompt, model_name=MODEL_NAME)
                # 查找数字 attitude
                return response
            except Exception as e:
                print(f"Error occurred: {e}")
        # 如果多次尝试都失败，保持原有态度不变
        return None
