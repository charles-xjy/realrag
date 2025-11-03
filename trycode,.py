from pathlib import Path
datas_dir = "./"
datas_dir = Path(datas_dir)
# output_base_dir = Path(output_base_dir)
pdf_files = list(datas_dir.rglob("*.pdf"))
if not pdf_files:
    print(f"未找到PDF文件于: {datas_dir}")
else:
    print(f"找到{len(pdf_files)}个PDF文件于: {datas_dir}\n")
for pdf_path in pdf_files:
    # 直接打印 WindowsPath 对象，会自动输出完整路径字符串
    print(pdf_path)