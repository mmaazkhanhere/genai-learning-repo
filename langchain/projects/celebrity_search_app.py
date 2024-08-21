import streamlit as st
from dotenv import load_dotenv, find_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

load_dotenv(find_dotenv())


st.title('Celebrity Search App') # title of the application

input = st.text_input("Search for a celebrity") # user input

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                            temperature=0.8,
                            
                            )

# create an instance of model. The temperature is set to 0.8 to make it creative in response

first_input_prompt = PromptTemplate(
    input_variables=["name"],
    template="Tell me about {name}"
) # a prompt template where name will be replaced by the input of the user

chain = LLMChain(llm=llm, prompt=first_input_prompt, output_key='person', verbose=True ) # a chain is created the runs the first prompt through the model and store the ouput in key 'person'

second_input_prompt = PromptTemplate(
    input_variables=["person"],
    template="In which year, {person} was born?"
) # second prompt that takes an input (input from previous response)

chain2 = LLMChain(llm=llm, prompt=second_input_prompt, output_key='dob', verbose=True)
# another chain is created that takes the output from the previous chain as input and runs the second prompt through the model. The output is stored in key 'dob'

third_input_prompt = PromptTemplate(
    input_variables=["dob"],
    template="Mention 5 major events that happened around {dob} in the world?"
) # another prompt template that takes an input dob (from the previous response)


chain3 = LLMChain(llm=llm,
                prompt=third_input_prompt,
                output_key='events',
                verbose=True)

# a chain is created that takes output from previous chain as input and run the third prompt into model whose ouput is stored in key 'events'

seq = SequentialChain(chains=[chain, chain2, chain3], verbose=True, input_variables=['name'], output_variables=['person', 'dob', 'events'])

# the 3 chains are combined in one chain. Output of one chain feeds to next allowing step by step input retrieval

if input: # displays information only when user input a name
    st.write(seq({'name': input}))