import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from langchain_core.documents import Document


def _inline_to_text(node: Dict[str, Any]) -> str:
    node_type = node.get("type")
    if node_type == "text":
        return node.get("text", "")
    if node_type == "equation":
        return f"${node.get('latex', '')}$"
    if node_type == "link":
        label = node.get("label") or node.get("id") or ""
        return f"{label}"
    return ""


def rich_text_to_plain_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return _block_to_text(value)
    if not isinstance(value, list):
        return str(value)

    lines = []
    for block in value:
        text = _block_to_text(block)
        if text:
            lines.append(text)
    return "\n".join(lines)


def _block_to_text(block: Dict[str, Any]) -> str:
    block_type = block.get("type")

    if "content" in block:
        text = "".join(_inline_to_text(node) for node in block.get("content") or [])
        if block_type == "heading":
            level = int(block.get("level") or 2)
            return f"{'#' * max(1, min(level, 6))} {text}"
        if block_type == "bulleted_list_item":
            return f"- {text}"
        if block_type == "numbered_list_item":
            number = block.get("number")
            prefix = f"{number}. " if number else "1. "
            return f"{prefix}{text}"
        if block_type == "blockquote":
            return f"> {text}"
        if block_type == "callout":
            icon = block.get("icon") or "提示"
            return f"{icon}: {text}"
        return text

    if block_type == "code_block":
        language = block.get("language", "")
        return f"```{language}\n{block.get('code', '')}\n```"
    if block_type == "equation_block":
        return f"$$\n{block.get('latex', '')}\n$$"
    if block_type == "image":
        return f"[图片: {block.get('alt') or block.get('blob_id') or ''}]"
    if block_type == "ref_card":
        label = block.get("label") or block.get("id") or ""
        link_type = block.get("link_type") or "reference"
        return f"[{link_type}: {label}]"
    if block_type == "divider":
        return "---"
    return ""


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _doc(content: str, metadata: Dict[str, Any]) -> Document:
    return Document(page_content=content.strip(), metadata={k: v for k, v in metadata.items() if v is not None})


def _course_docs(path: Path) -> List[Document]:
    payload = _load_json(path)
    course = payload.get("course", payload)
    course_id = course.get("key") or path.stem
    title = course.get("title") or course_id

    parts = [
        f"课程：{title}",
        f"课程简介：{rich_text_to_plain_text(course.get('summary'))}",
        rich_text_to_plain_text(course.get("content")),
    ]
    content = "\n\n".join(part for part in parts if part.strip())
    return [
        _doc(
            content,
            {
                "source_type": "course_content",
                "source_id": course_id,
                "title": title,
                "course_id": course_id,
                "graph_key": course.get("graph_key"),
            },
        )
    ]


def _knowledge_graph_docs(path: Path) -> List[Document]:
    payload = _load_json(path)
    graph = payload.get("graph", {})
    graph_name = graph.get("name") or graph.get("key") or path.stem
    docs = []

    graph_description = rich_text_to_plain_text(graph.get("description"))
    if graph_description:
        docs.append(
            _doc(
                f"知识图谱：{graph_name}\n\n{graph_description}",
                {
                    "source_type": "knowledge_graph",
                    "source_id": graph.get("key") or path.stem,
                    "title": graph_name,
                },
            )
        )

    for concept in payload.get("concepts", []):
        concept_id = concept.get("key") or concept.get("id")
        name = concept.get("name") or concept_id
        tags = concept.get("tags") or []
        description = rich_text_to_plain_text(concept.get("description"))
        content = "\n\n".join(
            part
            for part in [
                f"知识点：{name}",
                f"标签：{'、'.join(tags)}" if tags else "",
                description,
            ]
            if part.strip()
        )
        if content.strip():
            docs.append(
                _doc(
                    content,
                    {
                        "source_type": "concept",
                        "source_id": concept_id,
                        "title": name,
                        "graph_key": graph.get("key"),
                        "tags": "、".join(tags),
                    },
                )
            )
    return docs


def _problem_docs(path: Path) -> List[Document]:
    payload = _load_json(path)
    problems = payload.get("problems", [])
    docs = []
    for problem in problems:
        problem_id = problem.get("key") or problem.get("id")
        title = problem.get("title") or problem_id
        tags = problem.get("tags") or []
        concepts = problem.get("concepts") or []
        content = "\n\n".join(
            part
            for part in [
                f"题目：{title}",
                f"题干：{rich_text_to_plain_text(problem.get('content'))}",
                f"答案：{rich_text_to_plain_text(problem.get('answer'))}",
                f"解析：{rich_text_to_plain_text(problem.get('solution'))}",
                f"标签：{'、'.join(tags)}" if tags else "",
                f"关联知识点：{'、'.join(concepts)}" if concepts else "",
            ]
            if part.strip()
        )
        if content.strip():
            docs.append(
                _doc(
                    content,
                    {
                        "source_type": "problem",
                        "source_id": problem_id,
                        "title": title,
                        "tags": "、".join(tags),
                        "concept_ids": "、".join(concepts),
                    },
                )
            )
    return docs


def load_json_materials(paths: Iterable[str]) -> List[Document]:
    docs: List[Document] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists() or path.suffix.lower() != ".json":
            continue

        if path.name == "course.json":
            docs.extend(_course_docs(path))
        elif path.name == "knowledge_graph.json":
            docs.extend(_knowledge_graph_docs(path))
        elif path.name == "problems.json":
            docs.extend(_problem_docs(path))

    return docs
