from typing import Any, Dict, List

from temporalio import activity

from qa_service import (
    generate_answer,
    retrieve_documents,
    save_assistant_message,
    save_user_message,
)


@activity.defn(name="save_user_message_activity")
def save_user_message_activity(payload: Dict[str, Any]) -> Dict[str, Any]:
    return save_user_message(payload["conversation_id"], payload["content"])


@activity.defn(name="retrieve_documents_activity")
def retrieve_documents_activity(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    return retrieve_documents(payload["content"], payload.get("retrieval"))


@activity.defn(name="generate_answer_activity")
def generate_answer_activity(payload: Dict[str, Any]) -> str:
    return generate_answer(payload["content"], payload.get("retrieved_docs", []))


@activity.defn(name="save_assistant_message_activity")
def save_assistant_message_activity(payload: Dict[str, Any]) -> Dict[str, Any]:
    return save_assistant_message(
        payload["conversation_id"],
        payload["answer"],
        payload.get("retrieved_docs", []),
    )
