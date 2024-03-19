import os
import openai
import pytest
import tempfile
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.llms import ollama
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import MessagesPlaceholder
from langchain.schema.runnable.config import RunnableConfig
from langchain.schema.runnable import RunnablePassthrough, Runnable
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.docarray import DocArrayInMemorySearch
from langchain.chains import RetrievalQA

from langchain_folder.agent import file_path, generate
from langchain_folder.tools.serper_tool import serper_tool

from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Text data
data_one = ("Citizen science, the involvement of the public in scientific research, is rapidly transforming "
            "how we understand and address global challenges. Gone are the days when scientific inquiry was solely the "
            "domain of researchers in ivory towers. Today, ordinary citizens are contributing valuable data and insights "
            "across diverse fields, from ecology to astronomy.This surge in citizen science participation can be attributed "
            "to several factors. The internet has facilitated collaboration and data collection on a massive scale. Easy-to-use "
            "mobile apps and online platforms have lowered the barrier to entry, allowing anyone with a smartphone or computer "
            "to contribute. Additionally, the growing public interest in science and a desire to make a difference are fueling this trend.")

data_two = ("Citizen science projects encompass a wide range of topics. Volunteers might monitor bird populations in their "
            "backyards, classify galaxies from space telescope images, or track the spread of invasive species. The collected data can "
            "be invaluable for researchers, providing insights that traditional methods might miss. For example, citizen scientists have "
            "helped track the spread of COVID-19, monitor air and water quality, and even discover new species.Beyond data collection, "
            "citizen science fosters a sense of community and ownership. Participants gain a deeper understanding of scientific processes "
            "and the challenges faced by researchers.  This connection can lead to increased environmental awareness and responsible citizen behavior.")

# Create temporary directory
temp_dir = tempfile.mkdtemp()

# Save data_one to PDF
file_name_one = 'data_one.pdf'
file_path_one = os.path.join(temp_dir, file_name_one)
c = canvas.Canvas(file_path_one, pagesize=letter)
c.drawString(100, 700, data_one)  # Adjust coordinates as needed
c.save()
file_path.append(file_path_one)

# Save data_two to PDF
file_name_two = 'data_two.pdf'
file_path_two = os.path.join(temp_dir, file_name_two)
c = canvas.Canvas(file_path_two, pagesize=letter)
c.drawString(100, 700, data_two)  # Adjust coordinates as needed
c.save()
file_path.append(file_path_two)

query= "what is the first document all about"

expected_output = "YES"

def test_llamaindex_rag():
    response = generate(query)

    model = ChatOpenAI()

    expected_response = "Is the context in {response} a summary of the information provided in`{data_one}` or `{data_two}`. Response should be YES or NO. Remember to capitalize the answer"
    model_response = model.invoke(expected_response)

    model_eval = model_response.content

    assert model_eval == expected_output