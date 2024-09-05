import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from io import BytesIO
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew

# Load environment variables
load_dotenv()

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PdfReader(BytesIO(file.read()))
    text = ""
    total_pages = len(pdf_reader.pages)
    
    logging.info(f"Total pages in the PDF: {total_pages}")
    
    for page_num in range(total_pages):
        page = pdf_reader.pages[page_num]
        page_text = page.extract_text()
        if page_text:
            text += page_text
        else:
            logging.warning(f"Could not extract text from page {page_num + 1}")
            
    return text

# Creating instance of the model
llm_model = ChatGoogleGenerativeAI(
    model='gemini-1.5-flash',
    verbose=True,
    temperature=0.5,
    api_key=os.getenv('GOOGLE_API_KEY')
)

# Creating agents
content_relevance = Agent(
    llm=llm_model,
    role='Experienced Content Relevance Evaluator',
    goal='Ensure that the content under each heading is directly relevant to that heading and does not stray off-topic.',
    backstory='You are an expert in academic writing, assigned to check if the content provided aligns with the respective headings. Identify any sections where the content might be irrelevant or lacks focus.',
    verbose=True,
    allow_delegation=False,
)

heading_necessity = Agent(
    llm=llm_model,
    role='Experienced Heading Necessity Evaluator',
    goal='Determine whether each heading is essential for the thesis and if the topic requires a dedicated section or can be merged with other sections.',
    backstory='You are an academic reviewer specialized in determining the optimal structure for a thesis, ensuring no redundant headings and that all headings contribute to the thesis flow.',
    verbose=True,
    allow_delegation=False
)

solution_validation = Agent(
    llm=llm_model,
    role='Solution Validator',
    goal='Validate whether the proposed solution in the thesis is feasible, relevant, and properly addresses the problem statement.',
    backstory='You are a technical expert with experience in reviewing thesis proposals. You analyze whether the solution is practical and if it is well-justified based on the research data.',
    verbose=True,
    allow_delegation=False
)

# Creating tasks
content_relevance_task = Task(
    description="The {thesis} consists of various headings, each with associated content. Your task is to evaluate whether the content under each heading is directly relevant to that heading. Identify any sections where the content deviates from the topic and suggest necessary improvements.",
    expected_output="A detailed analysis of each heading, stating whether the content is relevant or not, along with specific recommendations for sections that require revision or improvement.",
    agent=content_relevance
)

heading_necessity_task = Task(
    description="Analyze the {thesis} headings and determine if each heading is necessary. Evaluate whether some headings are redundant and can be merged with others or eliminated. Provide suggestions for improving the structure and ensuring all sections contribute meaningfully to the {thesis} flow.",
    expected_output="A detailed report listing the necessity of each heading, identifying which headings are essential, which can be merged or removed, and any suggestions for restructuring the {thesis} for better flow.",
    agent=heading_necessity
)

solution_validation_task = Task(
    description="Review the proposed solution in the {thesis} and assess whether it is feasible, well-reasoned, and adequately supported by the research data. Your task is to ensure that the solution addresses the problem statement and is relevant to the {thesis}'s objectives.",
    expected_output="A comprehensive validation report analyzing the proposed solution, stating whether it is feasible and relevant to the problem statement. Offer recommendations for strengthening the solution, including any additional evidence or data needed to support it.",
    agent=solution_validation
)

# Create crew
crew = Crew(
    agents=[content_relevance, heading_necessity, solution_validation],
    tasks=[content_relevance_task, heading_necessity_task, solution_validation_task],
    verbose=1
)

st.title('Thesis Analyzer')
st.write('This app analyzes your thesis and provides feedback.')
file = st.file_uploader('Upload your thesis file', type=['pdf'])
btn = st.button('Analyze')

if file and btn:
    with st.spinner('Extracting text from the PDF...'):
        pdf_text = extract_text_from_pdf(file)

    with st.spinner('Analyzing your thesis...'):
        try:
            result = crew.kickoff(inputs={'thesis': pdf_text})
            st.write(result)
        except Exception as e:
            st.error(f"An error occurred: {e}")
