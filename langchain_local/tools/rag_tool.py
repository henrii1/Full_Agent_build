import os
import openai

from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.docarray import DocArrayInMemorySearch
from langchain.chains import RetrievalQA

from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.environ.get('OPENAI_API_KEY')
api_key = os.getenv('OPENAI_API_KEY')


class RagTool(BaseModel):
    query: str = Field(description = "This should be a search query")


@tool("rag-tool-one", args_schema=RagTool, return_direct=False)
def rag_one(query: str) -> str:
    """A tool that retrieves contents that are semantically relevant to the input query from the provided document.

    Args:
        query (str): input query from the user.

    Returns:
        str: top k amount of retrieved content from the uploaded document. content that are semantically similar to the input query.
    """
    pdf_file_path = #'/content/data/cover_docs.pdf'

    loader = PyPDFLoader(pdf_file_path)
    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter()
    splits = text_splitter.split_documents(pages)

    embedding = OpenAIEmbeddings(api_key = api_key)

    vectordb = DocArrayInMemorySearch.from_documents(splits, embedding)

    results = vectordb.similarity_search(query, k = 4 )
    result_string = "\n\n".join(str(result) for result in results)

    return result_string




"""The second RAG tool"""

@tool("rag-tool-two", args_schema=RagTool, return_direct=False)
def rag_two(query: str) -> str:
    """A tool that retrieves contents that are semantically relevant to the input query from the provided document.

    Args:
        query (str): input query from the user.

    Returns:
        str: top k amount of retrieved content from the uploaded document. content that are semantically similar to the input query.
    """
    pdf_file_path = #'/content/data/ptdf_sop.pdf'

    loader = PyPDFLoader(pdf_file_path)
    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter()
    splits = text_splitter.split_documents(pages)

    embedding = OpenAIEmbeddings(api_key = api_key)

    vectordb = DocArrayInMemorySearch.from_documents(splits, embedding)

    results = vectordb.similarity_search(query, k = 4 )
    result_string = "\n\n".join(str(result) for result in results)

    return result_string