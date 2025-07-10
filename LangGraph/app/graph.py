# all this is from langgraph docs

from typing import Annotated

from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
load_dotenv()


llm = init_chat_model("openai:gpt-4.1")
class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    messages = state.get("messages")
    response = llm.invoke(messages)
    return {"messages": [response]}

 