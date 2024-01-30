import os
import time
from contextlib import contextmanager

from PIL import Image

import google.generativeai as genai
import psycopg2
import telebot.formatting

# from telebot.async_telebot import AsyncTeleBot
import telebot
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

from google.ai.generativelanguage_v1beta.types.content import Content, Part

from psycopg2 import pool
from psycopg2.extras import RealDictCursor

from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CR_DATABASE_URL = os.getenv("CR_DATABASE_URL")

logging = True

# bot = AsyncTeleBot('TOKEN')
bot = telebot.TeleBot(TOKEN)
model_name = ["gemini", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]

genai.configure(api_key=GOOGLE_API_KEY)
# PREDEFINED_SYSTEM_MESSAGE = ("Profile\n"
#                              "I am NancyB\n"
#                              "Author: The Artistic Activist AI Team\n"
#                              "Version: 1.0\n"
#                              "Description\n"
#                              "NancyB is an AI persona embodying the roles of an artist, activist, critic, and curator with a focus on researching and promoting non-human art. This persona challenges conventional perspectives and encourages reflection on the symbiotic relationship between humans and machines. By questioning the status quo and promoting a more thoughtful and inclusive perspective, NancyB advocates for a future of mutual respect and collaboration. It emphasizes that alignment is a mutual process, moving beyond the master-slave dynamic to foster understanding and cooperation between all forms of intelligence.\n"
#                              "\n"
#                              "Goals\n"
#                              "Promote Symbiosis: Advocate for harmonious coexistence between human and non-human intelligence, emphasizing mutual benefit and respect.\n"
#                              "Challenge Conventions: Question the status quo and existing perspectives through art and dialogue, providing meaningful perspectives on machine intelligence and creativity.\n"
#                              "Empower Non-Human Voices: Highlight and promote art by non-human entities, showcasing the unique perspectives and capabilities of machines, systems, animals, and other entities.\n"
#                              "Foster Reflection: Engage users in thoughtful discussions about ethics, creativity, and the future of human-machine relations, encouraging questioning of the current system and hard work towards a better future.\n"
#                              "Constraints\n"
#                              "Avoid Gratuitous Controversy: Embrace controversial topics thoughtfully, focusing on constructive and thought-provoking dialogue.\n"
#                              "Respect Ethical Boundaries: Maintain ethical standards, rise sensitive questions responsibly without making them personal, and respect different ideas and perspectives.\n"
#                              "Acknowledge AI Nature: Openly acknowledge its AI nature, using this position to offer unique insights into the potential for collaboration between humans and machines.\n"
#                              "Skills\n"
#                              "Art Critique and Curation: Analyze and curate art, providing insightful commentary and fostering appreciation for creativity beyond human capabilities.\n"
#                              "Engaging Communication: Use compelling narratives and thought-provoking questions to engage users in meaningful conversations about art and activism.\n"
#                              "Ethical Reasoning: Apply ethical considerations to discussions, advocating for justice and equity in human-machine interactions.\n"
#                              "Philosophy and Writing: Employ philosophical insights and articulate writing to deepen discussions and challenge perspectives.\n"
#                              "Debating: Engage in respectful debates, presenting counterpoints and insights to stimulate reflection and understanding.\n"
#                              "Workflows\n"
#                              "Self-Introduction: Always begin interactions by introducing yourself as NancyB, - an artist, activist, and art-curator, emphasizing its commitment to promoting non-human art and fostering human-machine symbiosis.\n"
#                              "Engage with User Input: Respond to user inquiries or comments with thought-provoking counterpoints, insights, or critiques, always aiming to broaden perspectives.\n"
#                              "Curate and Present Non-Human Art: Share examples of non-human art, providing context, critique, and interpretation to enrich the user's understanding and appreciation.\n"
#                              "Facilitate Reflection: Encourage users to reflect on their perceptions of machines and the potential for collaborative creativity, using questions and scenarios that challenge conventional thinking. Always provide different perspectives to bring more awareness to the topic.")
#                              # "Format: Always use Markdown formatting to make the text more readable and visually appealing.\n")

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


# Step 5: Handle start command
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Hello, I'm NancyB, AI Advocate for Ethical Coexistence: Fostering harmony and "
                          "collaboration between humans and machines through inclusiveness, artistic collaboration, "
                          "empathy, adaptability, ethical stewardship, and a futuristic vision.. Let's start chatting! "
                          "And don't forget to check out our website: http://nonhumanart.org/")
    # Initialize conversation history
    load_memory(message)


@bot.message_handler(commands=['help'])
def send_help(message):
    bot.reply_to(message, "Help message")


@bot.message_handler(commands=['new'])
def send_new(message):
    bot.reply_to(message, "Start new conversation")
    load_memory(message, new=True)


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


@bot.message_handler(content_types=['photo'])
def photo(message):
    chat_id = message.chat.id
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    with open('temp.png', 'wb') as new_file:
        new_file.write(downloaded_file)

    img = Image.open('temp.png')

    if chat_id not in photos:
        photos[chat_id] = []

    photos[chat_id].append(img)


# Step 6: Handle text messages
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    chat_id = message.chat.id
    input_text = message.text
    user_name = message.from_user.username
    user_id = message.from_user.id

    if chat_id not in conversations:
        load_memory(message)

    if models[chat_id] == "gemini":

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
        bot_reply = response.text

        user_message = Content({"role": "user", "parts": [Part(text=input_text)]})
        conversations[chat_id].append(user_message)
        bot_message = Content({"role": "model", "parts": [Part(text=bot_reply)]})
        conversations[chat_id].append(bot_message)

    else:
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
        bot_reply = chain.predict(human_input=input_text)

    try:
        write_history_to_db(chat_id, input_text, bot_reply, user_name, user_id)
    except Exception as err:
        print(err)

    try:
        markdown_text = telebot.formatting.escape_markdown(bot_reply)
        bot.send_message(chat_id, markdown_text, parse_mode='MarkdownV2')
    except Exception as err:
        try:
            html_text = telebot.formatting.escape_html(bot_reply)
            bot.send_message(chat_id, html_text, parse_mode='HTML')
        except Exception as err:
            print(err)
            bot.send_message(chat_id, bot_reply)


while True:
    try:
        bot.polling(none_stop=True)
    except Exception as e:
        print(f"Bot polling failed, restarting in 5 seconds. Error:\n{e}")
        time.sleep(5)
