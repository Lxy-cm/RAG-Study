"""
RAG 适配器层：参考资料查找器 + LLM 回答生成器

角色 B - 参考资料和回答生成适配人
提供两个核心接口：
  - ReferenceFinder.find(query, options) → List[ReferenceMaterial]
  - AnswerWriter.write(query, references) → str
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

# 设置 HuggingFace 国内镜像（在导入检索模块之前）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


# ==========================================
# 数据模型
# ==========================================

@dataclass
class ReferenceMaterial:
    """检索到的参考资料

    作为 LangChain Document 与上层业务之间的中转数据结构。
    """
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 1.0
    source_id: str = ""
    title: str = ""

    # ---- 与 LangChain Document 互转 ----

    def to_langchain_document(self) -> Document:
        """转换为 LangChain Document（供生成模块消费）"""
        return Document(page_content=self.content, metadata=self.metadata)

    @classmethod
    def from_langchain_document(
        cls, doc: Document, score: float = 1.0
    ) -> "ReferenceMaterial":
        """从 LangChain Document 创建 ReferenceMaterial"""
        meta = doc.metadata or {}
        return cls(
            content=doc.page_content,
            metadata=meta,
            score=score,
            source_id=meta.get("source_id", ""),
            title=meta.get("title") or meta.get("Chapter", ""),
        )


# ==========================================
# 参考资料查找器
# ==========================================

class ReferenceFinder:
    """RAG 参考资料查找器

    封装检索链路：意图路由 → 向量召回 → 交叉重排。
    通过 options["limit"] 控制返回数量。
    """

    def __init__(self):
        from rag_retrieve import MathRetriever
        self._retriever = MathRetriever()

    def find(
        self, query: str, options: Optional[Dict[str, Any]] = None
    ) -> List[ReferenceMaterial]:
        """根据用户问题检索相关资料

        Args:
            query: 用户输入的问题
            options: 可选配置字典
                - limit (int): 返回的最大引用数量，默认 5，范围 [1, 20]

        Returns:
            ReferenceMaterial 列表；无结果时返回空列表
        """
        from rag_generate import smart_retrieve

        options = options or {}
        limit = max(1, min(int(options.get("limit", 5)), 20))

        # 走智能路由检索
        docs = smart_retrieve(query, self._retriever)

        if not docs:
            return []

        # 截取 limit 条并转换为 ReferenceMaterial
        results: List[ReferenceMaterial] = []
        for doc in docs[:limit]:
            ref = ReferenceMaterial.from_langchain_document(doc)
            results.append(ref)

        return results


# ==========================================
# LLM 回答生成器
# ==========================================

class AnswerWriter:
    """LLM 回答生成器

    接收用户问题与检索到的参考资料，调用大模型生成最终回答。
    无参考资料时返回兜底回答。
    """

    def __init__(self):
        from rag_generate import init_llm
        self._llm = init_llm()

    def write(self, query: str, references: List[ReferenceMaterial]) -> str:
        """基于参考资料生成回答

        Args:
            query: 用户问题
            references: ReferenceMaterial 列表（可能为空）

        Returns:
            生成的回答字符串
        """
        # 空引用处理：兜底回答
        if not references:
            return "根据现有知识库，无法准确解答。"

        from rag_generate import generate_math_answer

        # ReferenceMaterial → LangChain Document
        docs = [ref.to_langchain_document() for ref in references]

        return generate_math_answer(query=query, retrieved_docs=docs, llm=self._llm)


# ==========================================
# 便捷函数（一键问答）
# ==========================================

def ask(query: str, limit: int = 5) -> Dict[str, Any]:
    """一键问答：检索 + 生成

    Args:
        query: 用户问题
        limit: 引用数量

    Returns:
        {"query": str, "references": List[ReferenceMaterial], "answer": str}
    """
    finder = ReferenceFinder()
    writer = AnswerWriter()

    refs = finder.find(query, options={"limit": limit})
    answer = writer.write(query, refs)

    return {"query": query, "references": refs, "answer": answer}
