import os
import openai
import tempfile


from langchain_core.tools import Tool, tool
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
from langchain.pydantic_v1 import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.docarray import DocArrayInMemorySearch
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.llms import ollama
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser


class RagTool(BaseModel):
    query: str = Field(description = "This should be a search query")

class SerperTool(BaseModel):
  query: str = Field(description = "This should be a search query")



"""refactor this class. you can't associate decorators with class methods"""
  
class langchain_agent:
    def __init__(self, openai_api_key: str,
                serper_api_key: str,
                file_one_path: str, 
                file_two_path: str):
        self.openai_api_key = openai_api_key
        self.serper_api_key = serper_api_key
        self.file_one_path = file_one_path
        self.file_two_path = file_two_path

    @tool("rag-tool-one", args_schema=RagTool, return_direct=False)
    def rag_one(self, query: str) -> str:
        """A tool that retrieves contents that are semantically relevant to the input query from the provided document.

        Args:
            query (str): input query from the user.

        Returns:
            str: top k amount of retrieved content from the uploaded document. content that are semantically similar to the input query.
        """
        try:
            pdf_file_path = self.file_one_path

            loader = PyPDFLoader(pdf_file_path)
            pages = loader.load_and_split()

            text_splitter = RecursiveCharacterTextSplitter()
            splits = text_splitter.split_documents(pages)

            embedding = OpenAIEmbeddings(api_key = self.openai_api_key)

            vectordb = DocArrayInMemorySearch.from_documents(splits, embedding)

            results = vectordb.similarity_search(query, k = 4 )
            result_string = "\n\n".join(str(result) for result in results)

        except FileNotFoundError:
            result_string = ""

        return result_string

    @tool("rag-tool-two", args_schema=RagTool, return_direct=False)
    def rag_two(self, query: str) -> str:
        """A tool that retrieves contents that are semantically relevant to the input query from the provided document.

        Args:
            query (str): input query from the user.

        Returns:
            str: top k amount of retrieved content from the uploaded document. content that are semantically similar to the input query.
        """
        try:
            pdf_file_path = self.file_two_path

            loader = PyPDFLoader(pdf_file_path)
            pages = loader.load_and_split()

            text_splitter = RecursiveCharacterTextSplitter()
            splits = text_splitter.split_documents(pages)

            embedding = OpenAIEmbeddings(api_key = self.openai_api_key)

            vectordb = DocArrayInMemorySearch.from_documents(splits, embedding)

            results = vectordb.similarity_search(query, k = 4 )
            result_string = "\n\n".join(str(result) for result in results)
        except FileNotFoundError:
            result_string = ""

        return result_string
    
    @tool("serper_tool_main", args_schema=SerperTool, return_direct=False)
    def serper_tool(self, query:str) -> str:
        """A useful for when you need to ask with search. Very useful when recent or specific information is needed from the web
        """
        search = GoogleSerperAPIWrapper(k=4, type="search", serper_api_key=self.serper_api_key)
        initial_result = search.results(query)
        result = initial_result['organic']
        results = ""
        for r in result:
            data = f"'Title':{r['title']}\n 'content':{r['snippet']}"
            results += f"{data}\n\n"
        return results
    
    def generate_langchain(self, query: str) -> str:

        tools = [self.rag_one, self.rag_two, self.serper_tool]

        # functions = [convert_to_openai_function(f) for f in tools]
        # model = ChatOpenAI(temperature= 0, api_key = api_key).bind_functions(functions=functions)

        model = ChatOpenAI(temperature= 0, api_key = self.openai_api_key).bind_tools(tools=tools)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert agent with the ability to decide if a function is needed and route queries to the right function. Don't ask, just route to the function"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        # agent_chain = RunnablePassthrough.assign(
        #         agent_scratchpad = lambda x: format_to_openai_functions(x["intermediate_steps"])
        # ) | prompt | model | OpenAIFunctionsAgentOutputParser()

        agent_chain = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | model
    | OpenAIToolsAgentOutputParser()
)
        memory = ConversationBufferMemory(return_messages = True, memory_key = "chat_history")

        agent_executor = AgentExecutor(agent = agent_chain, tools = tools, verbose= False, memory = memory)

        result = agent_executor.invoke({"input": query})
        return result['output']






def main():
    st.title("Multi-Agent Chatbot")
    st.write("Ask questions based on the uploaded document and get a response")

    texts = st.text_area("Enter your questions here")

    file_uploaded = st.sidebar.file_uploader(
        "Upload two PDF files containing any document", key="file_upload",
        accept_multiple_files = True,
        type=['pdf']
    )

    temp_dir = tempfile.mkdtemp()

    if st.sidebar.button("Upload PDF File"):
        if file_uploaded:
           for file in file_uploaded:
            file_dest = os.path.join(temp_dir, file.name)
            bytes_data = file.read()
            with open(file_dest, "wb") as f:
                f.write(bytes_data)
            #file_path.append(file_dest)
            
    if st.button("Ask Question"):
        with st.spinner(text="Generating"):
            pass
            #response = generate_langchain(texts)
            #st.markdown(response)

        


if __name__ =="__main__":
    import streamlit as st

    main()



