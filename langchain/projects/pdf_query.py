import streamlit as st
import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv, find_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

# Load environment variables
load_dotenv(find_dotenv())

# Initialize the model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.8)

# Streamlit app
st.title('PDF Query')
st.write('Upload a PDF file and enter your query to get the relevant information from the document.')

# File uploader
file = st.file_uploader('Upload a PDF file', type='pdf', accept_multiple_files=False)

if file:
    query = st.text_input('Enter your query')
    btn = st.button('Get Response')

    if btn:
        st.write('Converting PDF...')
        
        # Initialize PdfReader
        reader = PdfReader(file)
        raw_text = ""
        
        # Extract text from each page
        for page in reader.pages:
            content = page.extract_text()
            if content:
                raw_text += content
        
        st.write('Splitting text...')
        text_splitter = CharacterTextSplitter(chunk_size=800,
                                            chunk_overlap=200,
                                            length_function=len,
                                            separator='\n')
        
        texts = text_splitter.split_text(raw_text)

        st.write('Embedding texts...')
        embeddings = OpenAIEmbeddings()

        # Embed the texts and store in FAISS index
        document_search = FAISS.from_texts(texts, embeddings)

        chain = load_qa_chain(model, chain_type="stuff")

        # Perform similarity search and get response
        docs = document_search.similarity_search(query)
        response = chain.run(input_documents=docs, question=query)

        st.write(response)
