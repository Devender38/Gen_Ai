import gradio as gr
from huggingface_hub import InferenceClient
from huggingface_hub import login

hf_token = "enter_Your_hugging_face_Token"
login(token = hf_token)

client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

def respond(message, history):
    system_message = "You are a helpful AI assistant."
    messages = [{"role": "system", "content": system_message}]

    for user_msg, bot_reply in history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if bot_reply:
            messages.append({"role": "assistant", "content": bot_reply})

    messages.append({"role": "user", "content": message})

    response = ""
    for message in client.chat_completion(
        messages, max_tokens=512, stream=True, temperature=0.7, top_p=0.95
    ):
        token = message.choices[0].delta.content
        response += token
        yield response

# Custom Styling
css = """
/* Responsive Layout */
@media (min-width: 1024px) {
    .chatbot-container {
        max-width: 70%;
        margin: auto;
    }
}

@media (max-width: 1023px) {
    .chatbot-container {
        max-width: 90%;
        margin: auto;
    }
}

.chatbot-container {
    background-color: #000000;
    color: #fff;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.3);
}
.message {
    border-radius: 50px;
    padding: 12px 18px;
    margin: 10px 0;
    font-size: 16px;
    font-weight: bold;
}
.user {
    background-color: #424242;
    color: black !important;
    text-align: right;
    border-radius: 50px;
    padding: 14px;
}
.bot {
    background-color: #424242;
    color: white;
    text-align: left;
    border-radius: 50px;
    padding: 14px;
}
.typing {
    font-style: italic;
    color: #bbb;
}

/* Gradient Animated Send Button */
button {
    background: linear-gradient(45deg, #00ff9c, #33707a, #f600ff);
    background-size: 300% 300%;
    color: white !important;
    font-weight: bold !important;
    border-radius: 50px !important;
    padding: 12px 18px !important;
    transition: 0.3s ease-in-out !important;
    animation: gradientMove 4s infinite alternate;
    border: none;
}

@keyframes gradientMove {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

button:hover {
    transform: scale(1.05);
}

/* Footer Styling */
.footer {
    text-align: center;
    font-size: 16px;
    font-weight: bold;
    padding: 15px;
    margin-top: 20px;
    background: linear-gradient(45deg, #bcd5ae, #06f417);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("# AI STUDY ASSISTANT")
    chatbot = gr.ChatInterface(respond)

    # Footer
    gr.Markdown("""
    <div class='footer'>
        <span style='background: linear-gradient(45deg, #bcd5ae, #06f417); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Created by Devender , Harsh</span>
    </div>
    """)

if __name__ == "__main__":
    demo.launch(share=True,pwa=True,debug=False)