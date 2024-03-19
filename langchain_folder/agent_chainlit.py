import os
import openai
import tempfile
import chainlit as cl


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
        pdf_file_path = ''
  
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
        pdf_file_path = ''

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

#model = ChatOllama(model="mistral").bind(functions = functions)



welcome_message = """Welcome to this AI agent that lets you chat with two large PDF files and 
also connects to the internet whenever recent information is needed. To get started:
1. Upload two PDFs 
2. Ask a question about the files or any random question requiring new information.
"""
@cl.on_chat_start
async def on_chat_start():
        await cl.Message(content="""Hello there, Welcome to this multi-tool Langchain Agent that connects to the internet and let's you chat
                         with your data""").send()
        files = None
        while files == None:
                files = await cl.AskFileMessage(
                        content="please upload a PDF file if you want to chat with a document, upload a max of two files", 
                        accept=["application/pdf"], max_files=2,
                        max_size_mb=20, timeout=180
                ).send()

        for file in files:

                # with tempfile.NamedTemporaryFile() as temp_file:
                #         temp_file.write(file.content)
                #         temp_file_path = temp_file.name
                while len(file_path) > 2:
                        file_path.pop(0)
                file_path.append(file)



        model = ChatOpenAI(temperature= 0, streaming=True).bind_functions(functions=functions)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert agent with the ability to decide if a function is needed and route queries to the right function"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{question}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        memory = ConversationBufferMemory(return_message = True, memory_key = "chat_history")
        agent_chain = RunnablePassthrough.assign(
                agent_scratchpad = lambda x: format_to_openai_functions(x["intermediate_steps"])
        ) | prompt | model | OpenAIFunctionsAgentOutputParser()

        agent_executor = AgentExecutor(agent = agent_chain, tools = tools, verbose= False, memory = memory)

        cl.user_session.set("runnable", agent_executor)


@cl.on_message
async def on_message(message: cl.Message):
        runnable = cl.user_session.get("runnable")

        msg = cl.Message(content="")

        async for chunk in runnable.astream(
                {"question": message.content,},
                config = RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),

        ):
                await msg.stream_token(chunk)
        
        await msg.send()








