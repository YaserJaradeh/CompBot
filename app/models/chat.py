from enum import Enum
from typing import Dict, Optional, Union
from uuid import UUID

from app.models.common import Request, Response


class AgentKind(Enum):
    """
    The type of agent to create.
    """

    JSON = "json"
    DATAFRAME = "dataframe"


class ChatRequest(Request):
    comparison_id: str
    message: str
    model: Optional[str] = "text-davinci-003"
    agent: Optional[AgentKind] = AgentKind.DATAFRAME


class ChatResponse(Response):
    # TODO: Customize Responses based on the type of message
    chat_id: UUID
    answer: Union[str, Dict]
