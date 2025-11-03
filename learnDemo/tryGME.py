from modelscope import AutoModel
from transformers import BitsAndBytesConfig
from transformers.utils.versions import require_version
import torch

if __name__ == "__main__":
    require_version(
        "transformers<4.52.0",
        "The remote code has some issues with transformers>=4.52.0, please downgrade: pip install transformers==4.51.3",
    )

    t2i_prompt = "Find an image that matches the given text."
    texts = [
        "The Tesla Cybertruck is a battery electric pickup truck built by Tesla, Inc. since 2023.",
        "Alibaba office.",
    ]
    images = [
    r'D:\code\learn dl\realrag\learnDemo\TaobaoCity_Alibaba_Xixi_Park.jpg',
    r'learnDemo/Tesla_Cybertruck_damaged_window.jpg',
    ]


    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 启用4比特加载
        bnb_4bit_quant_type="nf4",  # 使用NF4量化类型
        llm_int8_enable_fp32_cpu_offload=True,  # 关键：允许CPU上的模块用FP32精度
        bnb_4bit_use_double_quant=True,  # 启用双重量化
        bnb_4bit_compute_dtype=torch.bfloat16,  # 设置矩阵乘法的计算数据类型
    )
    gme = AutoModel.from_pretrained(
        "iic/gme-Qwen2-VL-2B-Instruct",
        torch_dtype="float16",
        device_map="cuda",
        trust_remote_code=True,
        # quantization_config=quantization_config,
    )

# 添加异常处理
    try:
        # Single-modal embedding
        e_text = gme.get_text_embeddings(texts=texts)
        e_image = gme.get_image_embeddings(images=images)
        print("Single-modal", (e_text @ e_image.T).tolist())
    except Exception as e:
        print(f"Error occurred: {e}")
        # 可以在这里添加备用逻辑，比如使用本地图像或跳过某些图像
        
    # How to set embedding instruction
    e_query = gme.get_text_embeddings(texts=texts, instruction=t2i_prompt)
    # If is_query=False, we always use the default instruction.
    e_corpus = gme.get_image_embeddings(images=images, is_query=False)
    print('Single-modal with instruction', (e_query @ e_corpus.T).tolist())
    ## Single-modal with instruction [[0.429931640625, 0.11505126953125], [0.049835205078125, 0.409423828125]]

    # Fused-modal embedding
    e_fused = gme.get_fused_embeddings(texts=texts, images=images)
    print('Fused-modal', (e_fused @ e_fused.T).tolist())
    ## Fused-modal [[1.0, 0.05511474609375], [0.05511474609375, 1.0]]
