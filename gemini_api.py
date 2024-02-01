import os
from google.ai.generativelanguage_v1beta.types.content import Content, Part
import google.generativeai as genai

from dotenv import load_dotenv

from db_logging import get_history_from_db
from messages import GEMINI_SYSTEM_MESSAGE

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 8192,
}


def handle_gemini_model(chat_id, input_text, conversations, photos):
    if chat_id in photos and photos[chat_id]:
        try:
            img = photos[chat_id].pop()
            prompt = GEMINI_SYSTEM_MESSAGE + "user input: " + input_text
            model = genai.GenerativeModel('gemini-pro-vision')
            response = model.generate_content(contents=[prompt, img])
            response.resolve()
        except Exception as err:
            print(err)
            model = genai.GenerativeModel('gemini-pro', generation_config=generation_config)
            chat = model.start_chat(history=conversations[chat_id])
            response = chat.send_message(input_text)
    else:
        model = genai.GenerativeModel('gemini-pro', generation_config=generation_config)
        chat = model.start_chat(history=conversations[chat_id])
        response = chat.send_message(input_text)
    return response.text


def load_gemini_memory(conversation, chat_id):
    if len(conversation) == 0:
        user_message = Content({"role": "user", "parts": [Part(text="hi")]})
        conversation.append(user_message)
        bot_message = Content({"role": "model", "parts": [Part(text=GEMINI_SYSTEM_MESSAGE)]})
        conversation.append(bot_message)

        history = get_history_from_db(chat_id)
        # reverse history
        history.reverse()
        for hist in history:
            user_message = Content({"role": "user", "parts": [Part(text=hist["question"])]})
            conversation.append(user_message)
            bot_message = Content({"role": "model", "parts": [Part(text=hist["answer"])]})
            conversation.append(bot_message)

    return conversation
