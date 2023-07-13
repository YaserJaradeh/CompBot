from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from app.core.utils import log
from app.models.chat import ChatRequest
from app.services.comparison_chat import ComparisonChatService
from app.services.ws.manager import ConnectionManager

router = APIRouter(prefix="/chat", tags=["chat"])
manager = ConnectionManager()


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


@router.websocket("/stream")
@log(__name__)
async def ws_stream(
    websocket: WebSocket,
):
    service = ComparisonChatService()
    await manager.connect(websocket)
    try:
        while True:
            request = await websocket.receive_json()
            await service.query_agent_ws(
                comparison_id=request["comparison_id"],
                query=request["message"],
                model=request["model"],
                agent_kind=request["agent"],
                websocket=websocket,
                ws_manager=manager,
            )
    except WebSocketDisconnect:
        manager.disconnect(websocket)
