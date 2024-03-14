import os
import pytest

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.agents.format_scratchpad import format_to_openai_functions
from dotenv import load_dotenv, find_dotenv


from tools.rag_tool import rag_one, rag_two
from tools.serper_tool import serper_tool

_ = load_dotenv(find_dotenv())

api_key = os.getenv('OPENAI_API_KEY')

path_one = '/workspaces/Full_Agent_build/cover_docs.pdf'
path_two = 
