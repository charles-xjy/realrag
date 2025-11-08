from unittest import result
from modelscope import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tomlkit import item

from modelscope import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


def AsyncImageAnalysis(path, title_max_length="10", description_max_length="100"):
    # default: Load the model on the available device(s)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="sdpa",
    )

    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2-VL-2B-Instruct",
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )

    # default processer
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

    # The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
    # min_pixels = 256*28*28
    # max_pixels = 1280*28*28
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": path,
                },
                {
                    "type": "text",
                    "text": f"请分析这张图片并生成一个{title_max_length}字以上的标题、{description_max_length}字以上的图片描述，使用JSON格式输出。\n"
                    "分析以下方面:\n"
                    "1. 图像类型（图表、示意图、照片等）\n"
                    "2. 主要内容/主题\n"
                    "3. 包含的关键信息点\n"
                    "4. 图像的可能用途\n"
                    "\n"
                    "输出格式必须严格为:\n"
                    "{{\n"
                    f'"title": "标题(' + str(title_max_length) + '字以内)",\n'
                    f'"description": "详细描述('
                    + str(description_max_length)
                    + '字以内)"\n'
                    "}}\n"
                    "\n"
                    "只返回JSON,不要有其他说明文字。",
                },
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    # print(output_text)
    return output_text


import json

path = "/home/xjy/mycode/realrag/learnDemo/TaobaoCity_Alibaba_Xixi_Park.jpg"
a = AsyncImageAnalysis(path)
# print(a)
item = {}
if a:
    b = json.loads(a[0])
    item["image_caption"] = b["description"]
    item["image_title"] = b["title"]
# 生成一段包含图片和标题的 Markdown 格式文本
md = f"![{item}]({path})\n"
# print(md + "\n")
result = md + "\n"
# print(item)
# # 解析返回的JSON内容
# result_content = a[0]  # 提取列表中的JSON字符串
# parsed_result = json.loads(result_content)

# # 创建结构化JSON对象
# output_data = {
#     "filename": "1.png",
#     "path": path,
#     "title": parsed_result["title"],
#     "description": parsed_result["description"]
# }

# 保存为格式化的JSON文件
with open("learnDemo/output2.md", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
