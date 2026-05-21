from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, validator


class RetrievalExpansion(BaseModel):
    mode: Literal["USER_ONLY", "USER_PREFERRED"] = "USER_PREFERRED"


class RetrievalConfig(BaseModel):
    limit: int = Field(default=5, ge=1, le=20)
    query: Optional[Dict[str, Any]] = None
    expansion: RetrievalExpansion = Field(default_factory=RetrievalExpansion)


class SendMessageRequest(BaseModel):
    content: str
    retrieval: Optional[RetrievalConfig] = None

    @validator("content")
    def content_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("content 不能为空")
        return value


class CreateConversationRequest(BaseModel):
    title: Optional[str] = None


class ConversationResponse(BaseModel):
    conversation_id: str
    title: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


class Citation(BaseModel):
    citation_id: str
    source_type: str
    source_id: str
    title: Optional[str] = None
    snippet: Optional[str] = None
    score: float = 1.0


class MessageResponse(BaseModel):
    message_id: str
    role: Literal["user", "assistant"]
    content: str
    citations: List[Citation] = Field(default_factory=list)
    created_at: datetime


class HistoryMessageResponse(MessageResponse):
    feedback_rating: Optional[str] = None
    feedback_comment: Optional[str] = None
