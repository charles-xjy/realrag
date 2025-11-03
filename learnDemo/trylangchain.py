from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredPDFLoader

file_path = "./1.pdf"
loader = PyPDFLoader(file_path)


# 加载扫描件 PDF，启用 OCR（依赖 Tesseract，需提前安装配置）
loader = UnstructuredPDFLoader(
    file_path="./1.pdf",
    strategy="fast",  # 快速模式，适合纯文本扫描件
    extract_images=True,  # 不需要提取图片时设为 False
    ocr_language="chi_sim",  # 中文识别，英文用 "eng"
)
docs = loader.load()
docs[0]
print(docs[0].page_content)
