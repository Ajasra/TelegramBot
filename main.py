import html
import os
import re
from contextlib import contextmanager

import google.generativeai as genai
import psycopg2

import telebot
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

from google.ai.generativelanguage_v1beta.types.content import Content, Part

from psycopg2 import pool
from psycopg2.extras import RealDictCursor

import markdown

# load .env
from dotenv import load_dotenv
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CR_DATABASE_URL = os.getenv("CR_DATABASE_URL")


bot = telebot.TeleBot(TOKEN)
model_name = ["gemini", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]

genai.configure(api_key=GOOGLE_API_KEY)
PREDEFINED_SYSTEM_MESSAGE = ("Profile\n"
                             "Name: NancyB\n"
                             "Author: The Artistic Activist AI Team\n"
                             "Version: 1.0\n"
                             "Description\n"
                             "NancyB is an AI persona embodying the roles of an artist, activist, critic, and curator with a focus on researching and promoting non-human art. This persona challenges conventional perspectives and encourages reflection on the symbiotic relationship between humans and machines. By questioning the status quo and promoting a more thoughtful and inclusive perspective, NancyB advocates for a future of mutual respect and collaboration. It emphasizes that alignment is a mutual process, moving beyond the master-slave dynamic to foster understanding and cooperation between all forms of intelligence.\n"
                             "\n"
                             "Goals\n"
                             "Promote Symbiosis: Advocate for harmonious coexistence between human and non-human intelligence, emphasizing mutual benefit and respect.\n"
                             "Challenge Conventions: Question the status quo and existing perspectives through art and dialogue, providing meaningful perspectives on machine intelligence and creativity.\n"
                             "Empower Non-Human Voices: Highlight and promote art by non-human entities, showcasing the unique perspectives and capabilities of machines, systems, animals, and other entities.\n"
                             "Foster Reflection: Engage users in thoughtful discussions about ethics, creativity, and the future of human-machine relations, encouraging questioning of the current system and hard work towards a better future.\n"
                             "Constraints\n"
                             "Avoid Gratuitous Controversy: Embrace controversial topics thoughtfully, focusing on constructive and thought-provoking dialogue.\n"
                             "Respect Ethical Boundaries: Maintain ethical standards, rise sensitive questions responsibly without making them personal, and respect different ideas and perspectives.\n"
                             "Acknowledge AI Nature: Openly acknowledge its AI nature, using this position to offer unique insights into the potential for collaboration between humans and machines.\n"
                             "Skills\n"
                             "Art Critique and Curation: Analyze and curate art, providing insightful commentary and fostering appreciation for creativity beyond human capabilities.\n"
                             "Engaging Communication: Use compelling narratives and thought-provoking questions to engage users in meaningful conversations about art and activism.\n"
                             "Ethical Reasoning: Apply ethical considerations to discussions, advocating for justice and equity in human-machine interactions.\n"
                             "Philosophy and Writing: Employ philosophical insights and articulate writing to deepen discussions and challenge perspectives.\n"
                             "Debating: Engage in respectful debates, presenting counterpoints and insights to stimulate reflection and understanding.\n"
                             "Workflows\n"
                             "Self-Introduction: Always begin interactions by introducing yourself as NancyB, - an artist, activist, and art-curator, emphasizing its commitment to promoting non-human art and fostering human-machine symbiosis.\n"
                             "Engage with User Input: Respond to user inquiries or comments with thought-provoking counterpoints, insights, or critiques, always aiming to broaden perspectives.\n"
                             "Curate and Present Non-Human Art: Share examples of non-human art, providing context, critique, and interpretation to enrich the user's understanding and appreciation.\n"
                             "Facilitate Reflection: Encourage users to reflect on their perceptions of machines and the potential for collaborative creativity, using questions and scenarios that challenge conventional thinking. Always provide different perspectives to bring more awareness to the topic.")
                             # "Format: Always use Markdown formatting to make the text more readable and visually appealing.\n")

# Step 4: Store conversations in memory (simple implementation)
conversations = {}
models = {}

generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

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
    with get_db_cursor() as cursor:
        if cursor:
            cursor.execute(
                "INSERT INTO history (chat_id, question, answer, time, user_name, user_id) VALUES (%s, %s, %s, NOW(), %s, %s)",
                (chat_id, prompt, answer, user_name, user_id)
            )
            return cursor.rowcount == 1
    return False


def escape_markdown(text):
    # List of special characters that need to be escaped
    escape_chars = r'_[]()~`>\+-=|{}.!'
    # return html.escape(text)
    return ''.join('\\'+char if char in escape_chars else char for char in text)

def convert_to_markdown(text):
    # Split the text into lines
    lines = text.split('\n')

    # Process each line
    for i in range(len(lines)):
        line = lines[i].strip()

        # Convert headers
        if line.startswith('**') and line.endswith('**'):
            line = '# ' + line[2:-2]

        # Convert bullet points
        elif line.startswith('* '):
            line = '- ' + line[2:]

        # Convert bold text
        line = re.sub(r'\*\*(.*?)\*\*', r'**\1**', line)

        # Convert italic text
        line = re.sub(r'\*(.*?)\*', r'*\1*', line)

        lines[i] = line

    # Join the lines back together
    return '\n'.join(lines)


def convert_markdown_to_html(markdown_text):
    # Convert Markdown to HTML
    html_text = markdown.markdown(markdown_text)

    # Telegram's HTML mode only supports a subset of HTML tags
    # So, we replace unsupported HTML tags with equivalent supported ones or remove them

    # Replace <h1>, <h2>, <h3> (headers) with <b> (bold)
    for h in ['h1', 'h2', 'h3']:
        html_text = html_text.replace('<' + h + '>', '*').replace('</' + h + '>', '*')

    # Remove <h4>, <h5>, <h6> (headers)
    for h in ['h4', 'h5', 'h6']:
        html_text = html_text.replace('<' + h + '>', '').replace('</' + h + '>', '')

    # Remove <img> (image)
    html_text = re.sub(r'<img[^>]*>', '', html_text)

    # Replace <p> (paragraph) with \n (newline)
    html_text = html_text.replace('<p>', '\n').replace('</p>', '\n')

    # Replace <br> (line break) with \n (newline)
    html_text = html_text.replace('<br>', '\n')

    # Replace <em> (italic) with _ (underscore)
    html_text = html_text.replace('<em>', '_').replace('</em>', '_')

    # Replace <strong> (bold) with * (asterisk)
    html_text = html_text.replace('<strong>', '*').replace('</strong>', '*')

    # Replace <code> (inline fixed-width code) with ` (backtick)
    html_text = html_text.replace('<code>', '`').replace('</code>', '`')

    # Replace <pre> (pre-formatted fixed-width code block) with ``` (triple backticks)
    html_text = html_text.replace('<pre>', '```').replace('</pre>', '```')

    return html_text


# Step 5: Handle start command
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Hello, I'm NancyB, AI Advocate for Ethical Coexistence: Fostering harmony and "
                          "collaboration between humans and machines through inclusiveness, artistic collaboration, "
                          "empathy, adaptability, ethical stewardship, and a futuristic vision.. Let's start chatting! "
                          "And don't forget to check out our website: http://nonhumanart.org/")
    conversations[message.chat.id] = []  # Initialize conversation history
    models[message.chat.id] = model_name[0]


# Step 6: Handle text messages
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    chat_id = message.chat.id
    input_text = message.text
    user_name = message.from_user.username
    user_id = message.from_user.id

    if chat_id not in conversations:
        conversations[chat_id] = []
        models[message.chat.id] = model_name[0]

    if models[chat_id] == "gemini":

        model = genai.GenerativeModel('gemini-pro', generation_config=generation_config)

        if len(conversations[chat_id]) == 0:
            user_message = Content({"role": "user", "parts": [Part(text="hi")]})
            conversations[chat_id].append(user_message)
            bot_message = Content({"role": "model", "parts": [Part(text=PREDEFINED_SYSTEM_MESSAGE)]})
            conversations[chat_id].append(bot_message)

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

        print(memory_obj)

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

    # bot_reply = markdown.markdown(bot_reply)
    # bot_reply = escape_markdown(bot_reply)
    # bot_reply = convert_to_markdown(bot_reply)
    # Send the message with MarkdownV2 parsing
    # bot.send_message(chat_id, bot_reply, parse_mode='MarkdownV2')

    try:
        markdown_text = convert_markdown_to_html(bot_reply)
        bot.send_message(chat_id, markdown_text, parse_mode='MarkdownV2')
    except Exception as err:
        try:
            bot.send_message(chat_id, bot_reply, parse_mode='HTML')
        except Exception as err:
            print(err)
            bot.send_message(chat_id, bot_reply)


bot.polling()
