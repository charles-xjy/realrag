research_topic = "白杨淀的水为什么是绿色的"
prompt = f"""
            你是一个研究规划专家。你的任务是将用户的研究主题分解为3-5个子任务。

            
            研究主题：{research_topic}

            请分析这个研究主题，将其分解为3-5个子任务。每个子任务应该：
            1. 涵盖主题的一个重要方面
            2. 有明确的研究目标
            3. 可以通过搜索引擎找到相关资料

            请以JSON格式返回子任务列表，每个子任务包含：
            - title：任务标题（简洁明了）
            - intent：任务意图（为什么要研究这个）
            - query：搜索查询（用于搜索引擎的查询字符串，可以使用英文以获得更好的搜索结果）

            示例输出：
            [
            {{
                "title": "什么是多模态模型",
                "intent": "了解多模态模型的基础概念，为后续研究打下基础",
                "query": "multimodal model definition concept 2024"
            }},
            ...
            ]

            请确保：
            1. 子任务数量在3-5个之间
            2. 子任务之间有逻辑关系（如从基础到应用，从现状到趋势）
            3. 搜索查询能够准确找到相关资料
            4. 只返回JSON，不要包含其他文本
            """
import requests
import os

token = os.getenv("SILICONFLOW_API_KEY")
print(token)
url = "https://api.siliconflow.cn/v1/chat/completions"

payload = {
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": prompt}],
}
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

response = requests.post(url, json=payload, headers=headers)

print(response.json().get("choices")[0].get("message").get("content"))
