import asyncio
import os

from temporalio.client import Client
from temporalio.worker import Worker

from qa_activities import (
    generate_answer_activity,
    retrieve_documents_activity,
    save_assistant_message_activity,
    save_user_message_activity,
)
from qa_workflows import QAWorkflow


async def main() -> None:
    temporal_address = os.getenv("TEMPORAL_ADDRESS", "localhost:7233")
    task_queue = os.getenv("TEMPORAL_TASK_QUEUE", "qa-task-queue")

    client = await Client.connect(temporal_address)
    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[QAWorkflow],
        activities=[
            save_user_message_activity,
            retrieve_documents_activity,
            generate_answer_activity,
            save_assistant_message_activity,
        ],
    )
    print(f"QA Temporal worker started. address={temporal_address}, task_queue={task_queue}")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
