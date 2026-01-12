import nbformat
import os

# 注意：请确保这个路径和你本地文件的实际路径一致
# 按照你之前的 commit 记录，文件应该在 notebooks/ 目录下
file_path = 'notebooks/fiqa_rag_context_optimization.ipynb'

if os.path.exists(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    if 'widgets' in nb.metadata:
        del nb.metadata['widgets']

    for cell in nb.cells:
        if 'metadata' in cell and 'widgets' in cell.metadata:
            del cell.metadata['widgets']

    with open(file_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"✅ {file_path} cleaned successfully.")
else:
    print(f"❌ can't find file: {file_path}, please check the path is correct.")