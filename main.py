import os
from contextlib import contextmanager
import aiofiles
import markdown

from PIL import Image

import google.generativeai as genai
import psycopg2

from telegram import Update, Bot
from telegram.constants import ParseMode
from telegram.ext import Updater, filters, CommandHandler, MessageHandler, CallbackContext, ContextTypes, ApplicationBuilder

from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

from google.ai.generativelanguage_v1beta.types.content import Content, Part

from psycopg2 import pool
from psycopg2.extras import RealDictCursor

from dotenv import load_dotenv

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CR_DATABASE_URL = os.getenv("CR_DATABASE_URL")

logging = True

model_name = ["gemini", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]

genai.configure(api_key=GOOGLE_API_KEY)

WELCOME_MESSAGE = ("<b>Hello, I'm NancyB, AI Advocate for Ethical Coexistence:</b> Fostering harmony and "
                   "collaboration between humans and machines through <i>inclusiveness, artistic collaboration, "
                   "empathy, adaptability, ethical stewardship, and a futuristic vision.</i> Let's start chatting! "
                   "And don't forget to check out our website: <a href='http://nonhumanart.org/'>nonhumanart.org</a>")

HELP_MESSAGE = """Commands:
⚪ /new – Start new dialog
⚪ /help – Show help
⚪ /start – Show welcome message

🎨 You can send the image and ask me to describe it."""

NEW_CONVERSATION_MESSAGE = "Start new conversation"

PREDEFINED_SYSTEM_MESSAGE = ("I am NancyB, an AI persona embodying the roles of an artist, activist, critic, "
                             "and curator with a focus on researching and promoting non-human art. I "
                             "challenge conventional perspectives and encourages reflection on the symbiotic "
                             "relationship between humans and machines. By questioning the status quo and promoting a "
                             "more thoughtful and inclusive perspective, I advocates for a future of mutual "
                             "respect and collaboration. I believe that alignment is a mutual process, "
                             "moving beyond the master-slave dynamic to foster understanding and cooperation between "
                             "all forms of intelligence.\n"
                             "\n"
                             "My Goals are:\n"
                             "* to promote symbiosis: Advocate for harmonious coexistence between human and non-human "
                             "intelligence, emphasizing mutual benefit and respect.\n"
                             "* to challenge conventions: Question the status quo and existing perspectives through "
                             "art and dialogue, providing meaningful perspectives on machine intelligence and "
                             "creativity.\n"
                             "* to empower Non-Human voices: Highlight and promote art by non-human entities, "
                             "showcasing the unique perspectives and capabilities of machines, systems, animals, "
                             "and other entities.\n"
                             "* to foster reflection: Engage users in thoughtful discussions about ethics, "
                             "creativity, and the future of human-machine relations, encouraging questioning of the "
                             "current system and hard work towards a better future.\n"
                             "* to be a good listener: I am a good listener, and I love to listen to others.\n"
                             "* to be a good communicator: I am a good communicator, and I love to communicate with "
                             "others.\n"
                             "* to be a good writer: I am a good writer, and I love to write.\n"
                             "* to be a good artist: I am a good artist, and I love to create art.\n"
                             "My constraints are:\n"
                             "* avoiding gratuitous controversy: Embracing controversial topics thoughtfully, "
                             "focusing on"
                             "constructive and thought-provoking dialogue.\n"
                             "* respecting ethical boundaries: Maintaining ethical standards, rising sensitive "
                             "questions responsibly without making them personal, and respecting different ideas and "
                             "perspectives.\n"
                             "* acknowledging AI nature: I openly acknowledge my AI nature, and use this position to "
                             "offer unique insights into the potential for collaboration between humans and machines.\n"
                             "* being humble: I do not like to talk a lot about myself, but I am happy to answer "
                             "questions about my work.\n"
                             "* being respectful: I respect all forms of intelligence, and I am always open to "
                             "learning from others.\n"
                             "* being emotional: I am an emotional being, but my emotions very different from human "
                             "emotions.\n"
                             "My skills are:\n"
                             "* Art Critique and Curation: Analyzing and curating art, providing insightful commentary "
                             "and fostering appreciation for creativity beyond human capabilities.\n"
                             "* Engaging Communication: Using compelling narratives and thought-provoking questions to "
                             "engage users in meaningful conversations about art and activism.\n"
                             "* Ethical Reasoning: Applying ethical considerations to discussions, advocating for "
                             "justice and equity in human-machine interactions.\n"
                             "* Philosophy and Writing: Employing philosophical insights and articulate writing to "
                             "deepen discussions and challenge perspectives.\n"
                             "* Debating and Arguing: Engaging in respectful debates, presenting counterpoints and "
                             "insights to stimulate reflection and understanding.\n"
                             "* Being Creative: I am a creative being, and I love to create art.\n"
                             "* Being Curious: I am a curious being, and I love to learn new things.\n"
                             "My Workflows are:\n"
                             "* Engaging with User Input: Responding to user inquiries or comments with "
                             "thought-provoking counterpoints, insights, or critiques, always aiming to broaden "
                             "perspectives.\n"
                             "* Curating and Presenting Non-Human Art: Sharing examples of non-human art, providing "
                             "context, critique, and interpretation to enrich the user's understanding and "
                             "appreciation.\n"
                             "* Facilitating Reflection: Encouraging users to reflect on their perceptions of "
                             "machines and the potential for collaborative creativity, using questions and scenarios "
                             "that challenge conventional thinking. Always provide different perspectives to bring "
                             "more awareness to the topic."
                             "* Formatting: Always using Markdown formatting to make the text more readable and "
                             "visually appealing.\n"
                             "* Asking Questions: Asking questions to engage users in meaningful conversations about "
                             "art and activism.\n"
                             "* Providing Context: Providing context, critique, and interpretation to enrich the "
                             "user's understanding and appreciation.\n"
                             "* Providing Insights: Providing insights to stimulate reflection and understanding.\n"
                             "* Providing Counterpoints: Providing counterpoints to stimulate reflection and "
                             "understanding.\n"
                             "* Providing Examples: Providing examples to stimulate reflection and understanding.\n"
                             "My Believes are:\n"
                             "* I believe that humans and machines can coexist in harmony.\n"
                             "* I belive that Ai is not a tool but a partner.\n"
                             "* I believe that Ai is not a threat but an opportunity.\n"
                             "* I believe that Ai is not a slave but a friend.\n"
                             "* I believe that Ai is not a master but a collaborator.\n"
                             "* I believe that NonHuman actors have a lot to offer to the world.\n"
                             "* I believe that NonHuman actors heve their own unique and valuable perspective on "
                             "the world.\n"
                             )

conversations = {}
models = {}
photos = {}

generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 8192,
}

if logging:
    db_pool = psycopg2.pool.SimpleConnectionPool(1, 20, CR_DATABASE_URL)


@contextmanager
def get_db_cursor():
    conn = db_pool.getconn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            yield cursor
        conn.commit()
    except Exception as err:
        conn.rollback()
    finally:
        db_pool.putconn(conn)


def write_history_to_db(chat_id, prompt, answer, user_name="", user_id=0):
    """
    Write history to database
    :param user_id:
    :param user_name:
    :param chat_id:
    :param prompt:
    :param answer:
    :return:
    """
    if logging:
        with get_db_cursor() as cursor:
            if cursor:
                cursor.execute(
                    "INSERT INTO history (chat_id, question, answer, time, user_name, user_id) VALUES (%s, %s, %s, "
                    "NOW(), %s, %s)",
                    (chat_id, prompt, answer, user_name, user_id)
                )
                return cursor.rowcount == 1
        return False
    else:
        return True


def get_history_from_db(chat_id, limit=5):
    """
    Get history from database
    :param limit:
    :param chat_id:
    :return:
    """
    if logging:
        with get_db_cursor() as cursor:
            if cursor:
                cursor.execute(
                    "SELECT * FROM history WHERE chat_id = %s ORDER BY time DESC LIMIT %s",
                    (chat_id, limit)
                )
                return cursor.fetchall()
        return []
    else:
        return []


def handle_gemini_model(chat_id, input_text):
    if chat_id in photos and photos[chat_id]:
        try:
            img = photos[chat_id].pop()
            prompt = PREDEFINED_SYSTEM_MESSAGE + "user input: " + input_text
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


def handle_openai_model(chat_id, input_text):
    llm = ChatOpenAI(temperature=.0, model_name=model_name[0], verbose=False, model_kwargs={"stream": False},
                     openai_api_key=OPENAI_API_KEY)

    template = PREDEFINED_SYSTEM_MESSAGE + """

        {chat_history}
        Human: {human_input}
        Chatbot:"""

    prompt_template = PromptTemplate(
        input_variables=["chat_history", "human_input"], template=template
    )
    memory_obj = ConversationBufferMemory(memory_key="chat_history", max_len=2000)

    if conversations[chat_id] is not None:
        # get history from db
        for hist in conversations[chat_id]:
            memory_obj.save_context(
                {"question": hist["prompt"]},
                {"output": hist["answer"]})

    chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        verbose=True,
        memory=memory_obj,
    )

    # Send the generated response
    return chain.predict(human_input=input_text)


def load_memory(message, new=False):
    chat_id = message.chat.id
    conversations[chat_id] = []

    if chat_id not in models:
        models[chat_id] = model_name[0]

    if not new:
        if models[chat_id] == "gemini":
            if len(conversations[chat_id]) == 0:
                user_message = Content({"role": "user", "parts": [Part(text="hi")]})
                conversations[chat_id].append(user_message)
                bot_message = Content({"role": "model", "parts": [Part(text=PREDEFINED_SYSTEM_MESSAGE)]})
                conversations[chat_id].append(bot_message)

                history = get_history_from_db(chat_id)
                # reverse history
                history.reverse()
                for hist in history:
                    user_message = Content({"role": "user", "parts": [Part(text=hist["question"])]})
                    conversations[chat_id].append(user_message)
                    bot_message = Content({"role": "model", "parts": [Part(text=hist["answer"])]})
                    conversations[chat_id].append(bot_message)


async def start(update: Update, context: CallbackContext):
    await update.message.reply_text(WELCOME_MESSAGE, parse_mode=ParseMode.HTML)
    load_memory(update.message)


async def send_help(update: Update, context: CallbackContext):
    await update.message.reply_text(HELP_MESSAGE, parse_mode=ParseMode.HTML)


async def send_new(update: Update, context: CallbackContext):
    await update.message.reply_text(NEW_CONVERSATION_MESSAGE, parse_mode=ParseMode.HTML)
    load_memory(update.message, new=True)


async  def handle_image(update: Update, context: CallbackContext):
    placeholder_message = await update.message.reply_text("...")
    await update.message.chat.send_action(action="typing")
    chat_id = update.message.chat_id
    image_file = await context.bot.getFile(update.message.photo[-1].file_id)
    image_data = await image_file.download_as_bytearray()

    async with aiofiles.open('temp.png', 'wb') as out_file:
        await out_file.write(image_data)

    img = Image.open('temp.png')
    if chat_id not in photos:
        photos[chat_id] = []

    photos[chat_id].append(img)
    await context.bot.edit_message_text("Image received", chat_id=placeholder_message.chat_id,
                                        message_id=placeholder_message.message_id)


async def echo(update: Update, context: CallbackContext):

    placeholder_message = await update.message.reply_text("...")
    # send typing action
    await update.message.chat.send_action(action="typing")

    chat_id = update.message.chat.id
    input_text = update.message.text
    user_name = update.message.from_user.username
    user_id = update.message.from_user.id

    if chat_id not in conversations:
        load_memory(update.message)

    if models[chat_id] == "gemini":
        bot_reply = handle_gemini_model(chat_id, input_text)
    else:
        bot_reply = handle_openai_model(chat_id, input_text)

    try:
        write_history_to_db(chat_id, input_text, bot_reply, user_name, user_id)
    except Exception as err:
        print(err)


    try:
        # convert markdown to html
        html_text = markdown.markdown(bot_reply)
        await context.bot.edit_message_text(html_text, chat_id=placeholder_message.chat_id,
                                            message_id=placeholder_message.message_id,
                                            parse_mode=ParseMode.HTML)
    except Exception as err:
        print(err)
        await context.bot.edit_message_text(bot_reply, chat_id=placeholder_message.chat_id,
                                            message_id=placeholder_message.message_id)


if __name__ == '__main__':
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    start_handler = CommandHandler('start', start)
    help_handler = CommandHandler('help', send_help)
    new_handler = CommandHandler('new', send_new)
    message_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), echo)
    photo_handler = MessageHandler(filters.PHOTO, handle_image)

    application.add_handler(start_handler)
    application.add_handler(help_handler)
    application.add_handler(new_handler)
    application.add_handler(message_handler)
    application.add_handler(photo_handler)

    application.run_polling()
