import json
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.routers import chat


def get_application():
    _app = FastAPI(title=settings.PROJECT_NAME)

    __configure_routers(_app)
    _configure_cors_policy(_app)
    _save_openapi_specification(_app)
    return _app


def _configure_cors_policy(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def __configure_routers(app):
    app.include_router(chat.router)


def _save_openapi_specification(app):
    app_dir = os.path.dirname(os.path.realpath(__file__))
    _write_json(app.openapi(), os.path.join(app_dir, "..", "openapi.json"))


def _write_json(data, input_path):
    with open(input_path, "w") as f:
        json.dump(data, f, indent=4)


app = get_application()
