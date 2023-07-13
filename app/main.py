# -*- coding: utf-8 -*-
from fastapi.responses import HTMLResponse

from app.app_factory import get_application

app = get_application()


# @app.get("/", response_class=HTMLResponse)
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


@app.get("/")
def index():
    return HTMLResponse(
        """
<html>
    <head>
        <title>Grizzly Chat</title>
    </head>
    <body>
        <h1>Chat with Grizzly via WebSockets</h1>
        <form action="" onsubmit="sendMessage(event)">
            <label for="comparisonText">Comparison:</label>
            <input type="text" id="comparisonText" autocomplete="off" value="R595154"/><br><br>

            <label for="messageText">Message:</label>
            <textarea type="text" id="messageText" rows="4" cols="50">
Which models are fine-tuned on human instructions?
            </textarea><br><br>

            <label for="agentSelect">Agent:</label>
            <select id="agentSelect">
                <option value="dataframe" selected>dataframe</option>
                <option value="json">json</option>
            </select><br><br>

            <label for="modelSelect">Model:</label>
            <select id="modelSelect">
                <option value="text-davinci-003" selected>text-davinci-003</option>
                <option value="gpt-3.5-turbo">gpt-3.5-turbo</option>
                <option value="gpt-3.5-turbo-16k">gpt-3.5-turbo-16k</option>
                <option value="gpt-4">gpt-4</option>
            </select><br><br>

            <button>Send</button>
        </form>
        <button onclick="clearMessages()">Clear output</button>
        <ul id='messages'>
        </ul>
        <script>
            var ws = new WebSocket("ws://localhost:4321/chat/stream");
            ws.onmessage = function(event) {
                var messages = document.getElementById('messages')
                var message = document.createElement('li')
                var content = document.createTextNode(event.data)
                message.appendChild(content)
                messages.appendChild(message)
            };
            function sendMessage(event) {
                var comparisonInput = document.getElementById("comparisonText");
                var messageInput = document.getElementById("messageText");
                var agentSelect = document.getElementById("agentSelect");
                var modelSelect = document.getElementById("modelSelect");

                var comparison = comparisonInput.value;
                var message = messageInput.value;
                var agent = agentSelect.value;
                var model = modelSelect.value;

                var payload = {
                    comparison_id: comparison,
                    message: message,
                    agent: agent,
                    model: model
                };

                ws.send(JSON.stringify(payload));

                event.preventDefault();
            }
            function clearMessages() {
                var messages = document.getElementById('messages');
                messages.innerHTML = "";
            }
        </script>
    </body>
</html>
    """
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=4321)
