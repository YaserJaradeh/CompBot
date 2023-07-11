from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class Response(BaseModel):
    timestamp: datetime
    uuid: UUID


class Request(BaseModel):
    pass
