import streamlit as st
import os
from dotenv import load_dotenv, find_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(find_dotenv())


st.title('Celebrity Search App')

input = st.text_input("Search for a celebrity")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.8, api_key="AIzaSyD_n-us9oc2YQ4Nh1xjaIoIjaMWfzI_9VU")

if input:
    st.write(llm.invoke(input))