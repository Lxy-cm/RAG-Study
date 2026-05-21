import os
import json
import uuid
from typing import AsyncIterator, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from qa_models import (
    ConversationResponse,
    CreateConversationRequest,
    HistoryMessageResponse,
    MessageResponse,
    SendMessageRequest,
)
from qa_service import (
    answer_question,
    create_new_conversation,
    get_existing_conversation,
    list_conversation_messages,
    list_existing_conversations,
    normalize_retrieval,
    prepare_stream_answer,
    save_assistant_message,
    stream_answer,
    temporal_enabled,
    temporal_required,
)


app = FastAPI(title="RAG Study QA API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/frontend")
def frontend() -> FileResponse:
    return FileResponse(os.path.join(os.path.dirname(__file__), "frontend_test.html"))


@app.post("/conversations", response_model=ConversationResponse)
def create_conversation_endpoint(request: CreateConversationRequest) -> dict:
    return create_new_conversation(request.title)


@app.get("/conversations", response_model=List[ConversationResponse])
def list_conversations_endpoint(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> List[dict]:
    return list_existing_conversations(limit=limit, offset=offset)


@app.get("/conversations/{conversation_id}", response_model=ConversationResponse)
def get_conversation_endpoint(conversation_id: str) -> dict:
    conversation = get_existing_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="conversation 不存在")
    return conversation


async def _run_with_temporal(payload: dict) -> dict:
    from temporalio.client import Client

    temporal_address = os.getenv("TEMPORAL_ADDRESS", "localhost:7233")
    task_queue = os.getenv("TEMPORAL_TASK_QUEUE", "qa-task-queue")
    client = await Client.connect(temporal_address)
    handle = await client.start_workflow(
        "QAWorkflow",
        payload,
        id=f"qa-{payload['conversation_id']}-{uuid.uuid4().hex}",
        task_queue=task_queue,
    )
    return await handle.result()


async def _answer_with_best_available_runtime(payload: dict) -> dict:
    if temporal_enabled():
        try:
            return await _run_with_temporal(payload)
        except Exception:
            if temporal_required():
                raise

    return answer_question(
        conversation_id=payload["conversation_id"],
        content=payload["content"],
        retrieval=payload.get("retrieval"),
    )


@app.post("/conversations/{conversation_id}/messages", response_model=MessageResponse)
async def send_message(conversation_id: str, request: SendMessageRequest) -> dict:
    payload = {
        "conversation_id": conversation_id,
        "content": request.content,
        "retrieval": normalize_retrieval(request.retrieval),
    }

    try:
        return await _answer_with_best_available_runtime(payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/conversations/{conversation_id}/messages/stream")
async def send_message_stream(conversation_id: str, request: SendMessageRequest) -> StreamingResponse:
    async def event_stream() -> AsyncIterator[str]:
        full_content = ""
        try:
            retrieval = normalize_retrieval(request.retrieval)
            retrieved_docs = prepare_stream_answer(conversation_id, request.content, retrieval)
            for chunk in stream_answer(request.content, retrieved_docs):
                full_content += chunk
                yield f"event: message\ndata: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"

            saved_message = save_assistant_message(conversation_id, full_content, retrieved_docs)
            yield f"event: citations\ndata: {json.dumps({'citations': saved_message.get('citations', [])}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'detail': str(exc)}, ensure_ascii=False)}\n\n"
        finally:
            yield "event: done\ndata: {\"done\": true}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/conversations/{conversation_id}/messages", response_model=List[HistoryMessageResponse])
def get_messages(
    conversation_id: str,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> List[dict]:
    return list_conversation_messages(conversation_id, limit=limit, offset=offset)
