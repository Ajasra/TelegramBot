import os

from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

from messages import OPENAI_SYSTEM_MESSAGE

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def handle_openai_model(chat_id, input_text, conversations, model_name):
    llm = ChatOpenAI(temperature=.0, model_name=model_name, verbose=False, model_kwargs={"stream": False},
                     openai_api_key=OPENAI_API_KEY)

    template = OPENAI_SYSTEM_MESSAGE + """

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
