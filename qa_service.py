import os
import uuid
from threading import Lock
from typing import Any, Dict, Iterator, List, Optional

from qa_models import RetrievalConfig
from qa_storage import create_conversation, get_conversation, list_conversations, create_message, list_messages


_RAG_LOCK = Lock()
_RETRIEVER_ENGINE = None
_LLM = None


def _get_rag_runtime():
    global _LLM, _RETRIEVER_ENGINE

    with _RAG_LOCK:
        if _RETRIEVER_ENGINE is None or _LLM is None:
            from rag_generate import init_llm
            from rag_retrieve import MathRetriever

            _RETRIEVER_ENGINE = MathRetriever()
            _LLM = init_llm()

    return _RETRIEVER_ENGINE, _LLM


def save_user_message(conversation_id: str, content: str) -> Dict[str, Any]:
    return create_message(conversation_id, "user", content)


def create_new_conversation(title: Optional[str] = None) -> Dict[str, Any]:
    return create_conversation(title)


def get_existing_conversation(conversation_id: str) -> Optional[Dict[str, Any]]:
    return get_conversation(conversation_id)


def list_existing_conversations(limit: int, offset: int) -> List[Dict[str, Any]]:
    return list_conversations(limit=limit, offset=offset)


def _serialize_doc(doc: Any, index: int) -> Dict[str, Any]:
    metadata = getattr(doc, "metadata", {}) or {}
    content = getattr(doc, "page_content", str(doc))
    return {
        "doc_id": f"doc_{index + 1}",
        "content": content,
        "metadata": metadata,
        "score": 1.0,
    }


def retrieve_documents(content: str, retrieval: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    from rag_generate import smart_retrieve

    retriever_engine, _ = _get_rag_runtime()
    docs = smart_retrieve(content, retriever_engine)

    limit = 5
    if retrieval:
        limit = int(retrieval.get("limit") or limit)
    limit = max(1, min(limit, 20))

    return [_serialize_doc(doc, index) for index, doc in enumerate(docs[:limit])]


class _RetrievedDoc:
    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.page_content = content
        self.metadata = metadata


def generate_answer(content: str, retrieved_docs: List[Dict[str, Any]]) -> str:
    from rag_generate import generate_math_answer

    _, llm = _get_rag_runtime()
    docs = [
        _RetrievedDoc(doc.get("content", ""), doc.get("metadata", {}))
        for doc in retrieved_docs
    ]
    return generate_math_answer(query=content, retrieved_docs=docs, llm=llm)


def stream_answer(content: str, retrieved_docs: List[Dict[str, Any]]) -> Iterator[str]:
    if not retrieved_docs:
        yield "根据现有知识库，无法准确解答。"
        return

    from rag_generate import build_math_answer_prompt

    _, llm = _get_rag_runtime()
    docs = [
        _RetrievedDoc(doc.get("content", ""), doc.get("metadata", {}))
        for doc in retrieved_docs
    ]
    prompt_template, payload = build_math_answer_prompt(content, docs)
    messages = prompt_template.format_messages(**payload)

    for chunk in llm.stream(messages):
        text = getattr(chunk, "content", None)
        if text:
            yield text


def _source_id_from_metadata(metadata: Dict[str, Any], fallback: str) -> str:
    if metadata.get("source_id"):
        return str(metadata["source_id"])

    parts = [
        str(metadata.get(key)).strip()
        for key in ("Chapter", "Section", "Topic", "Sub_Topic")
        if metadata.get(key)
    ]
    return " / ".join(parts) if parts else fallback


def build_citations(retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    citations = []
    for index, doc in enumerate(retrieved_docs):
        metadata = doc.get("metadata", {}) or {}
        content = doc.get("content", "")
        source_id = _source_id_from_metadata(metadata, doc.get("doc_id", f"doc_{index + 1}"))
        title = metadata.get("title") or source_id
        citations.append(
            {
                "citation_id": f"citation_{uuid.uuid4().hex[:12]}",
                "source_type": str(metadata.get("source_type") or "course_content"),
                "source_id": source_id,
                "title": str(title) if title else None,
                "snippet": _build_snippet(content),
                "score": float(doc.get("score") or 1.0),
            }
        )
    return citations


def _build_snippet(content: str, max_length: int = 420) -> Optional[str]:
    if not content:
        return None

    snippet = content.strip()
    if len(snippet) <= max_length:
        return snippet

    snippet = snippet[:max_length].rstrip()
    if snippet.count("$$") % 2 == 1:
        last_math = snippet.rfind("$$")
        snippet = snippet[:last_math].rstrip()
    return f"{snippet}..."


def save_assistant_message(
    conversation_id: str,
    content: str,
    retrieved_docs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    citations = build_citations(retrieved_docs)
    return create_message(conversation_id, "assistant", content, citations)


def answer_question(
    conversation_id: str,
    content: str,
    retrieval: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    save_user_message(conversation_id, content)
    docs = retrieve_documents(content, retrieval)
    answer = generate_answer(content, docs)
    return save_assistant_message(conversation_id, answer, docs)


def prepare_stream_answer(
    conversation_id: str,
    content: str,
    retrieval: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    save_user_message(conversation_id, content)
    return retrieve_documents(content, retrieval)


def list_conversation_messages(conversation_id: str, limit: int, offset: int) -> List[Dict[str, Any]]:
    return list_messages(conversation_id, limit=limit, offset=offset)


def normalize_retrieval(retrieval: Optional[RetrievalConfig]) -> Optional[Dict[str, Any]]:
    if retrieval is None:
        return None

    if hasattr(retrieval, "model_dump"):
        return retrieval.model_dump()
    return retrieval.dict()


def temporal_enabled() -> bool:
    return os.getenv("QA_USE_TEMPORAL", "true").lower() not in {"0", "false", "no", "off"}


def temporal_required() -> bool:
    return os.getenv("QA_USE_TEMPORAL", "true").lower() == "required"
