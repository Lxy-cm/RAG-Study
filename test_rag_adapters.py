"""
rag_adapters.py 自测脚本

测试 ReferenceFinder 检索 和 AnswerWriter 生成 是否正常工作。
"""

from rag_adapters import ReferenceFinder, ReferenceMaterial, AnswerWriter


def test_reference_finder():
    """测试参考资料查找器"""
    print("=" * 50)
    print("测试 1: ReferenceFinder 检索")
    print("=" * 50)

    finder = ReferenceFinder()

    # 用默认 limit=5
    refs = finder.find("导数的定义是什么")

    print(f"\n检索到 {len(refs)} 条参考资料：\n")
    for i, ref in enumerate(refs):
        print(f"[{i + 1}] {ref.title}")
        print(f"    来源: {ref.source_id}")
        print(f"    内容摘要: {ref.content[:80]}...\n")

    assert len(refs) > 0, "应至少检索到 1 条参考资料"
    assert isinstance(refs[0], ReferenceMaterial), "返回类型应为 ReferenceMaterial"

    # 测试 limit 控制
    refs_3 = finder.find("导数的定义是什么", options={"limit": 3})
    assert len(refs_3) <= 3, f"limit=3 时返回数量应为 ≤3，实际为 {len(refs_3)}"

    print(f"limit=3 测试通过，实际返回 {len(refs_3)} 条\n")


def test_empty_references():
    """测试空引用场景"""
    print("=" * 50)
    print("测试 2: AnswerWriter 空引用兜底")
    print("=" * 50)

    writer = AnswerWriter()
    answer = writer.write("测试问题", [])

    print(f"\n空引用回答: {answer}\n")
    assert len(answer) > 0, "空引用时应有兜底回答"


def test_full_pipeline():
    """测试完整检索+生成流程"""
    print("=" * 50)
    print("测试 3: 检索 → 生成 完整流程")
    print("=" * 50)

    finder = ReferenceFinder()
    writer = AnswerWriter()

    query = "请解释导数的几何意义"
    refs = finder.find(query, options={"limit": 3})
    print(f"\n检索到 {len(refs)} 条参考资料")

    answer = writer.write(query, refs)
    print(f"\n回答:\n{answer}\n")

    assert len(answer) > 50, "回答不应过短"


def test_document_conversion():
    """测试 ReferenceMaterial ↔ LangChain Document 互转"""
    print("=" * 50)
    print("测试 4: Document 转换")
    print("=" * 50)

    from langchain_core.documents import Document

    # ReferenceMaterial → Document
    ref = ReferenceMaterial(
        content="这是测试内容",
        metadata={"source_id": "test_001", "Chapter": "第一章"},
        score=0.95,
        source_id="test_001",
        title="测试章节",
    )
    doc = ref.to_langchain_document()
    assert doc.page_content == "这是测试内容"
    assert doc.metadata["source_id"] == "test_001"

    # Document → ReferenceMaterial
    doc2 = Document(page_content="转换测试", metadata={"title": "转换标题"})
    ref2 = ReferenceMaterial.from_langchain_document(doc2, score=0.8)
    assert ref2.content == "转换测试"
    assert ref2.title == "转换标题"
    assert ref2.score == 0.8

    print("Document 互转测试通过\n")


def main():
    print("\n" + "=" * 50)
    print("rag_adapters.py 自测开始")
    print("=" * 50 + "\n")

    try:
        test_reference_finder()
        test_empty_references()
        test_full_pipeline()
        test_document_conversion()

        print("=" * 50)
        print("全部测试通过！")
        print("=" * 50)

    except Exception as e:
        print(f"\n测试失败: {e}")
        raise


if __name__ == "__main__":
    main()
