import os
import openai
import tempfile


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

from dotenv import load_dotenv
load_dotenv()

from langchain_folder.tools.serper_tool import serper_tool
api_key = os.getenv('OPENAI_API_KEY')
os.environ["SERPER_API_KEY"] = os.getenv('SERPER_API_KEY')




file_path = []

class RagToolOne(BaseModel):
    query: str = Field(description = "This should be a search query")


@tool("rag-tool-one", args_schema=RagToolOne, return_direct=False)
def rag_one(query: str) -> str:
    """A tool that retrieves contents that are semantically relevant to the input query from the provided document.

    Args:
        query (str): input query from the user.

    Returns:
        str: top k amount of retrieved content from the uploaded document. content that are semantically similar to the input query.
    """
    try:
        pdf_file_path = file_path[0]
  
        loader = PyPDFLoader(pdf_file_path)
        pages = loader.load_and_split()

        text_splitter = RecursiveCharacterTextSplitter()
        splits = text_splitter.split_documents(pages)

        embedding = OpenAIEmbeddings(api_key = api_key)

        vectordb = DocArrayInMemorySearch.from_documents(splits, embedding)

        results = vectordb.similarity_search(query, k = 4 )
        result_string = "\n\n".join(str(result) for result in results)
      
    except FileNotFoundError:
      result_string = ""

    return result_string




"""The second RAG tool"""
class RagToolTwo(BaseModel):
    query: str = Field(description = "This should be a search query")


@tool("rag-tool-two", args_schema=RagToolTwo, return_direct=False)
def rag_two(query: str) -> str:
    """A tool that retrieves contents that are semantically relevant to the input query from the provided document.

    Args:
        query (str): input query from the user.

    Returns:
        str: top k amount of retrieved content from the uploaded document. content that are semantically similar to the input query.
    """
    try:
        pdf_file_path = file_path[1]

        loader = PyPDFLoader(pdf_file_path)
        pages = loader.load_and_split()

        text_splitter = RecursiveCharacterTextSplitter()
        splits = text_splitter.split_documents(pages)

        embedding = OpenAIEmbeddings(api_key = api_key)

        vectordb = DocArrayInMemorySearch.from_documents(splits, embedding)

        results = vectordb.similarity_search(query, k = 4 )
        result_string = "\n\n".join(str(result) for result in results)
    except FileNotFoundError:
        result_string = ""

    return result_string


tools = [rag_one, rag_two, serper_tool]

functions = [convert_to_openai_function(f) for f in tools]
model = ChatOpenAI(temperature= 0, api_key = api_key).bind_functions(functions=functions)
#model = ChatOllama(model="mistral").bind(functions = functions)

def generate(query: str):
        """Function for interracting with the AI Agent"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert agent with the ability to decide if a function is needed and route queries to the right function"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        agent_chain = RunnablePassthrough.assign(
                agent_scratchpad = lambda x: format_to_openai_functions(x["intermediate_steps"])
        ) | prompt | model | OpenAIFunctionsAgentOutputParser()

        memory = ConversationBufferMemory(return_message = True, memory_key = "chat_history")

        agent_executor = AgentExecutor(agent = agent_chain, tools = tools, verbose= False, memory = memory)

        result = agent_executor.invoke({"input": query})
        return result['output']


def main():
    st.title("Multi-Agent Chatbot")
    st.write("Ask questions based on the uploaded document and get a response")

    texts = st.text_area("Enter your questions here")

    file_uploaded = st.sidebar.file_uploader(
        "Upload two PDF files containing any document", key="file_upload",
        accept_multiple_files = True
    )

    if st.sidebar.button("Upload PDF File"):
        if file_uploaded:
           for file in file_uploaded:
                while len(file_path) > 2:
                    file_path.pop(0)
                file_path.append(file.name)
            
    if st.button("Ask Question"):
        with st.spinner(text="Generating"):
            response = generate(texts)
            st.markdown(response)
        

if __name__ =="__main__":
    import streamlit as st

    main()

