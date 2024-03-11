import os

from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.docarray import DocArrayInMemorySearch
from langchain.chains import RetrievalQA

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # reading local files

openai_api_key = os.environ["OPENAI_API_KEY"]


class RagTool(BaseModel):
    query: str = Field(description = "This should be a search query")


@tool("rag-tool", args_schema=RagTool, return_direct=True)
def rag(query: str) -> str:
    """A tool that retrieves contents that are semantically relevant to the input query from the provided document.

    Args:
        query (str): input query from the user.

    Returns:
        str: top k amount of retrieved content from the uploaded document. content that are semantically similar to the input query.
    """
    pdf_file_path = _ #placehloder code from html, the path to pdf file
    
    loader = PyPDFLoader(pdf_file_path)
    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter()
    splits = text_splitter.split_documents(pages)

    embedding = OpenAIEmbeddings(api_key = openai_api_key)

    vectordb = DocArrayInMemorySearch.from_documents(splits, embedding)

    result = vectordb.similarity_search(query, k = 4 )
    result_string = "".join(result)

    return result_string


