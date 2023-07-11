# -*- coding: utf-8 -*-
from fastapi.responses import HTMLResponse

from app.app_factory import get_application

app = get_application()


@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
        <head>
            <title> Grizzly: A chatbot that understands and talks about ORKG comparisons </title>
        </head>
        <body>
            Welcome to Grizzly! The official API for the ORKG comparison chatbot.
        </body>
    </html>
    """


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=4321)
