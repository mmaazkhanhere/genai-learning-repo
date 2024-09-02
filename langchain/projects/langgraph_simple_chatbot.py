import streamlit as st
from dotenv import load_dotenv

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_groq import ChatGroq

load_dotenv()

class State(TypedDict): # something that will be received by the every node as input
    messages: Annotated[list, add_messages] # add messages is a reduce that tells LangGraph to append new messages to the existing list of messages
    
    # State keys without Annotation will be overwritten by each update. storing most recent values


graph_builder = StateGraph(State) # State consists of schema as well as reducers function which specify how to apply updates to the state

llm = ChatGroq(model_name="llama3-70b-8192", temperature=0.1)

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# adding an entry point that tell graphs where to start its work each time we run it
graph_builder.add_edge(START, "chatbot")

graph_builder.add_node("chatbot", chatbot) # first argument is the unique node name
                                        # second argument is the function or object that will be called when this node is executed


# adding a finish point. Tells the graph that any time this node is run, you can exit
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

st.title('Simple Chatbot With LangGraph')

user_input = st.text_input("Type your message here")
if st.button("Send"):
    st.write("You: " + user_input)
    if user_input.lower() in ["quit", "exit", "q"]:
        st.write("Goodbye!")
    for event in graph.stream({"messages": ("user", user_input)}):
        for value in event.values():
            st.write("Assistant:", value["messages"][-1].content)