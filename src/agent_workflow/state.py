from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    user_input: str
    messages: Annotated[list,add_messages]