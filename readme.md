**对于文档处理，使用mineru模块，并利用vllm加速**

**uv pip install mineru[all] -i https://pypi.tuna.tsinghua.edu.cn/simple**

mineru_parse_pdf是mineru官方给出的示例，输出的内容可以修改函数**_process_output**进行调整

在运行时指定backend为**vlm-transformers**或者**vlm-vllm-engine(利用vllm加速)**：

parse_doc(doc_path_list, output_dir, backend="vlm-transformers")

默认用mineru的模型MinerU2.5-2509-1.2B

os.environ["MINERU_MODEL_SOURCE"] = "modelscope"，指定模型来源为modelscope

对于data_paration，使用**AsyncImageAnalysis**函数进行图片描述

这里使用的是**Qwen/Qwen2-VL-2B-Instruct**，利用pytorch自带的加速**sdpa**

