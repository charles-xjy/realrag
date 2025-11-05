# from modelscope.models import Model
# from modelscope.pipelines import pipeline
# from modelscope.utils.constant import Tasks
# from PIL import Image
# import requests

# model = Model.from_pretrained(
#     "damo/multi-modal_gemm-vit-large-patch14_generative-multi-modal-embedding"
# )
# p = pipeline(task='image_captioning')

# url = "http://clip-multimodal.oss-cn-beijing.aliyuncs.com/lingchen/demo/dogs.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# image_path = "tryoutput/vlm/images/e7a0ed2636bd88d454dcd27e6eddf84e2c88b52cc5e319caf65123e3264fd4f2.jpg"
# text = "dogs playing in the grass"

# # img_embedding = p.forward({'image': image})['img_embedding']
# # print('image embedding: {}'.format(img_embedding))

# # text_embedding = p.forward({'text': text})['text_embedding']
# # print('text embedding: {}'.format(text_embedding))

# result = p(image_path)
# print(result)

# from modelscope.pipelines import pipeline
# from modelscope.utils.constant import Tasks

# pipe = pipeline(task=Tasks.visual_question_answering, model="Qwen/Qwen2-VL-2B-Instruct")
# result = pipe(
#     "tryoutput/vlm/images/e7a0ed2636bd88d454dcd27e6eddf84e2c88b52cc5e319caf65123e3264fd4f2.jpg",'描述这张图片.'
# )
# print(result)


# from modelscope.pipelines import pipeline
# from modelscope.utils.constant import Tasks

# model_id = "damo/mplug_image-captioning_coco_base_en"
# input_caption = (
#     "https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/mPLUG/image_captioning.png"
# )

# pipeline_caption = pipeline(Tasks.image_captioning, model=model_id)
# result = pipeline_caption(input_caption)
# print(result)
