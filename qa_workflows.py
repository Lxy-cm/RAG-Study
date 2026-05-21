from datetime import timedelta
from typing import Any, Dict

from temporalio import workflow


@workflow.defn
class QAWorkflow:
    @workflow.run
    async def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        activity_timeout = timedelta(minutes=10)

        await workflow.execute_activity(
            "save_user_message_activity",
            payload,
            start_to_close_timeout=activity_timeout,
        )
        retrieved_docs = await workflow.execute_activity(
            "retrieve_documents_activity",
            payload,
            start_to_close_timeout=activity_timeout,
        )
        answer = await workflow.execute_activity(
            "generate_answer_activity",
            {
                "content": payload["content"],
                "retrieved_docs": retrieved_docs,
            },
            start_to_close_timeout=activity_timeout,
        )
        return await workflow.execute_activity(
            "save_assistant_message_activity",
            {
                "conversation_id": payload["conversation_id"],
                "answer": answer,
                "retrieved_docs": retrieved_docs,
            },
            start_to_close_timeout=activity_timeout,
        )
