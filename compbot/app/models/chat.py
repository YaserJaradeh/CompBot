from typing import Dict, Union
from uuid import UUID

from compbot.app.models.common import Request, Response


class ChatRequest(Request):
    comparison_id: str
    message: str


class ChatResponse(Response):
    # TODO: Customize Responses based on the type of message
    chat_id: UUID
    answer: Union[str, Dict]
