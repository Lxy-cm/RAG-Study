import glob
import os

from json_material_loader import load_json_materials
from rag_split import build_and_save_vectorstore, process_math_markdown


def main() -> None:
    data_dir = os.getenv("DATA_DIR", "./data")
    md_files = sorted(glob.glob(os.path.join(data_dir, "*.md")))
    json_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    if not md_files and not json_files:
        raise FileNotFoundError(f"未在 {data_dir} 目录下找到 .md 或 .json 教材资料")

    all_docs = []
    for md_file in md_files:
        print(f"正在导入: {md_file}")
        all_docs.extend(process_math_markdown(md_file))

    json_docs = load_json_materials(json_files)
    if json_docs:
        print(f"正在导入 JSON 教材资料，共 {len(json_docs)} 条文档")
        all_docs.extend(json_docs)

    if not all_docs:
        raise ValueError("没有从资料文件中解析出可导入的文档")

    build_and_save_vectorstore(all_docs)
    print(f"知识库构建完成，共导入 {len(all_docs)} 个父文档。")


if __name__ == "__main__":
    main()
