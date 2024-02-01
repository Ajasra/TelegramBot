import os
import aiofiles
import markdown

from PIL import Image

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import filters, CommandHandler, MessageHandler, CallbackContext, ApplicationBuilder

from dotenv import load_dotenv

from db_logging import write_history_to_db
from gemini_api import handle_gemini_model, load_gemini_memory
from helpers import format_text_to_html
from messages import WELCOME_MESSAGE, HELP_MESSAGE, NEW_CONVERSATION_MESSAGE
from openai_api import handle_openai_model

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

model_name = ["gemini", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]
conversations = {}
models = {}
photos = {}


def load_memory(message, new=False):
    chat_id = message.chat.id
    conversations[chat_id] = []
    if chat_id not in models:
        models[chat_id] = model_name[0]
    if not new:
        if models[chat_id] == "gemini":
            conversations[chat_id] = load_gemini_memory(conversations[chat_id], chat_id)


async def start(update: Update, context: CallbackContext):
    await update.message.reply_text(WELCOME_MESSAGE, parse_mode=ParseMode.HTML)
    load_memory(update.message)


async def send_help(update: Update, context: CallbackContext):
    await update.message.reply_text(HELP_MESSAGE, parse_mode=ParseMode.HTML)


async def send_new(update: Update, context: CallbackContext):
    await update.message.reply_text(NEW_CONVERSATION_MESSAGE, parse_mode=ParseMode.HTML)
    load_memory(update.message, new=True)


async def handle_image(update: Update, context: CallbackContext):
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
    await update.message.chat.send_action(action="typing")

    chat_id = update.message.chat.id
    input_text = update.message.text
    user_name = update.message.from_user.username
    user_id = update.message.from_user.id

    if chat_id not in conversations:
        load_memory(update.message)

    if models[chat_id] == "gemini":
        bot_reply = handle_gemini_model(chat_id, input_text, conversations, photos)
    else:
        bot_reply = handle_openai_model(chat_id, input_text, conversations, models[chat_id])

    try:
        write_history_to_db(chat_id, input_text, bot_reply, user_name, user_id)
    except Exception as err:
        print(err)

    try:
        html_text = format_text_to_html(bot_reply)
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
