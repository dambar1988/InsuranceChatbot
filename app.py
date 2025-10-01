# app.py
from flask import Flask, request, jsonify, render_template_string
from backend import generate_answer

app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>TashaUI Insurance RAG Chatbot</title>
<style>
body {
    margin: 0;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #f5f5f5;
}
header {
    background-color: #343541;
    color: white;
    padding: 15px;
    text-align: center;
    font-size: 1.5em;
}
#chat-container {
    max-width: 800px;
    margin: 20px auto;
    background-color: #ffffff;
    border-radius: 10px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    display: flex;
    flex-direction: column;
    height: 600px;
}
#chat {
    flex: 1;
    padding: 20px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 10px;
}
.message {
    max-width: 70%;
    padding: 10px 15px;
    border-radius: 15px;
    word-wrap: break-word;
}
.user {
    background-color: #007bff;
    color: white;
    align-self: flex-end;
    border-bottom-right-radius: 0;
}
.bot {
    background-color: #e5e5ea;
    color: black;
    align-self: flex-start;
    border-bottom-left-radius: 0;
}
#input-area {
    display: flex;
    border-top: 1px solid #ddd;
}
#question {
    flex: 1;
    padding: 15px;
    border: none;
    border-radius: 0 0 0 10px;
    font-size: 1em;
}
#send-btn {
    padding: 0 20px;
    border: none;
    background-color: #007bff;
    color: white;
    cursor: pointer;
    font-size: 1em;
    border-radius: 0 0 10px 0;
}
#send-btn:hover { background-color: #0056b3; }
#question:focus { outline: none; }
</style>
</head>
<body>

<header>🤖 Allianz RAG Chatbot</header>

<div id="chat-container">
    <div id="chat"></div>
    <div id="input-area">
        <input type="text" id="question" placeholder="Type your question..." onkeypress="checkEnter(event)">
        <button id="send-btn" onclick="send()">Send</button>
    </div>
</div>

<script>
const chat = document.getElementById("chat");
const questionInput = document.getElementById("question");

function appendMessage(sender, text) {
    const div = document.createElement("div");
    div.className = "message " + sender;
    div.innerText = text;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
}

async function send() {
    const q = questionInput.value.trim();
    if(!q) return;
    appendMessage("user", q);
    questionInput.value = "";
    
    try {
        const res = await fetch("/chat", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({question: q})
        });
        const data = await res.json();
        appendMessage("bot", data.answer);
    } catch (err) {
        appendMessage("bot", "Error connecting to server.");
    }
}

function checkEnter(event) {
    if(event.key === "Enter") {
        send();
    }
}
</script>

</body>
</html>
"""

@app.route("/")
def index():
    return HTML_PAGE

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get("question", "")
    answer, sources = generate_answer(question)
    return jsonify({"answer": answer, "sources": sources})

if __name__ == "__main__":
    print("Starting Allianz RAG Chatbot at http://localhost:5100/")
    app.run(host="0.0.0.0", port=5100, debug=True, use_reloader=False)
