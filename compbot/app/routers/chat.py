from fastapi import APIRouter

from compbot.app.core.utils import log
from compbot.app.services.comparison_chat import ComparisonChatService

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/generate", status_code=200)
@log(__name__)
def generate():
    service = ComparisonChatService()
    return service
