import logging
import os
import sys
import pytest
import tempfile
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().handlers = []
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    SimpleKeywordTableIndex
)
from llama_index.core import SummaryIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from langchain_openai.chat_models import ChatOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.tools import QueryEngineTool, RetrieverTool
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

from llamaindex.main import generate

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

Settings.llm = OpenAI(model="gpt-3.5-turbo-1106", temperature=0.2)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

temp_dir = tempfile.mkdtemp()

data = ("Citizen science, the involvement of the public in scientific research, is rapidly transforming how we "
        "understand and address global challenges. Gone are the days when scientific inquiry was solely the domain of "
        "researchers in ivory towers. Today, ordinary citizens are contributing valuable data and insights across diverse fields, "
        "from ecology to astronomy. This surge in citizen science participation can be attributed to several factors. The internet "
        "has facilitated collaboration and data collection on a massive scale. Easy-to-use mobile apps and online platforms have "
        "lowered the barrier to entry, allowing anyone with a smartphone or computer to contribute. Additionally, the growing public "
        "interest in science and a desire to make a difference are fueling this trend.")

file_name = 'data.pdf'
file_path = os.path.join(temp_dir, file_name)

# Create PDF
c = canvas.Canvas(file_path, pagesize=letter)
c.drawString(100, 700, data)  # Adjust coordinates as needed
c.save()

query= "what is the provided document all about"

expected_output = "YES"

def test_llamaindex_rag():
    response = generate(query, temp_dir)

    model = ChatOpenAI()

    expected_response = "Is the context in {response} a summary of the information provided in`{data}`. Response should be YES or NO. Remember to capitalize the answer"
    model_response = model.invoke(expected_response)

    model_eval = model_response.content

    assert model_eval == expected_output