from .graph import create_chat_graph
from dotenv import load_dotenv

from langgraph.checkpoint.mongodb import MongoDBSaver
import os

load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")

config = {"configurable": {"thread_id": "3"}}
 
def init():
    with MongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:
        graph_with_mongo = create_chat_graph(checkpointer=checkpointer)
    
        while True:
            user_input = input("> ")
            for event in graph_with_mongo.stream({ "messages": [{"role": "user", "content": user_input}] }, config, stream_mode="values"):
                if "messages" in event:
                        event["messages"][-1].pretty_print()

init()