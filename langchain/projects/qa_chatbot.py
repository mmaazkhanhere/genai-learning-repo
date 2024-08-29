import streamlit as st
from dotenv import load_dotenv, find_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

load_dotenv(find_dotenv())

st.set_page_config(page_title='Q&A Chabot', page_icon=':robot:')

st.header('Hey! Lets Chat')

if 'flow_messages' not in st.session_state:
    st.session_state['flow_messages'] = [
        SystemMessage(content="You are a comedian AI assistant.")
    ]
    

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                            temperature=0.8,
                            )

def get_response(question):
    st.session_state['flow_messages'].append(HumanMessage(content=question))
    answer = model(st.session_state['flow_messages'])
    st.session_state['flow_messages'].append(AIMessage(content=answer.content))

    return answer.content

input = st.text_input('Ask a question')
btn = st.button('Ask')
if btn:
    response = get_response(input)
    st.header('The Response is: ')
    st.write(response)


