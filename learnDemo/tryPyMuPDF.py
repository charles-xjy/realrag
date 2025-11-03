import fitz
from pdf2image import convert_from_path
import pytesseract
import os
from pathlib import Path

# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def load_pdfs_from_dir(dir_path):
    """加载文件夹中所有 PDF 文件"""
    pdf_files = []
    for file in Path(dir_path).glob("*.pdf"):  # 遍历所有 .pdf 文件
        if file.is_file():
            pdf_files.append(str(file))
    return pdf_files


# 示例：加载 ./pdfs 文件夹中的所有 PDF
pdf_dir = "./"
all_pdfs = load_pdfs_from_dir(pdf_dir)
print(f"发现 {len(all_pdfs)} 个 PDF 文件")


def extract_text_from_pdf(pdf_path):
    """提取单份 PDF 的文本（支持 OCR 扫描件）"""
    doc = fitz.open(pdf_path)
    all_text = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()  # 尝试直接提取文本

        # 如果文本为空，判断为扫描件，用 OCR 提取
        if not str(text).strip():
            # 将 PDF 页转为图片
            images = convert_from_path(
                pdf_path,
                first_page=page_num + 1,
                last_page=page_num + 1,
                dpi=300,  # 分辨率越高，OCR 越准（但速度慢）
            )
            # 对图片执行 OCR
            ocr_text = pytesseract.image_to_string(
                images[0], lang="chi_sim"
            )  # 中文需安装语言包
            all_text.append(
                {
                    "page": page_num + 1,
                    "text": ocr_text,
                    "source": pdf_path,
                    "is_ocr": True,
                }
            )
        else:
            all_text.append(
                {
                    "page": page_num + 1,
                    "text": text,
                    "source": pdf_path,
                    "is_ocr": False,
                }
            )

    doc.close()
    return all_text


# 批量处理所有 PDF，提取文本
all_docs = []
for pdf in all_pdfs:
    print(f"处理 PDF: {pdf}")
    pdf_texts = extract_text_from_pdf(pdf)
    all_docs.extend(pdf_texts)  # 汇总所有文档的文本
# print(all_docs)
import json


# 在原有代码处理完 all_docs 后添加以下代码
with open("extracted_texts.json", "w", encoding="utf-8") as f:
    json.dump(all_docs, f, ensure_ascii=False, indent=2)

print("保存结果到文件")
