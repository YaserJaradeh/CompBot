from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from compbot.app.core.utils import log
from compbot.app.models.chat import ChatRequest
from compbot.app.services.comparison_chat import ComparisonChatService

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/generate", status_code=200)
@log(__name__)
def generate(
    request: ChatRequest,
):
    service = ComparisonChatService()
    return service.query_agent(
        comparison_id=request.comparison_id,
        query=request.message,
        model=request.model,
        agent_kind=request.agent,
    )


@router.post("/stream")
@log(__name__)
def stream(
    request: ChatRequest,
):
    service = ComparisonChatService()
    return StreamingResponse(
        service.query_agent_async(
            comparison_id=request.comparison_id,
            query=request.message,
            model=request.model,
            agent_kind=request.agent,
        ),
        media_type="text/event-stream",
    )
