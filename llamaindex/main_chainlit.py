import logging
import openai
import os
import sys
import tempfile
import chainlit as cl

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
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.tools import QueryEngineTool, RetrieverTool
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

openai.api_key = os.getenv('OPENAI_API_KEY')

Settings.llm = OpenAI(model="gpt-3.5-turbo-1106", temperature=0.2, streaming=True)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

document_dir = ""
documents = SimpleDirectoryReader(document_dir).load_data()

Settings.chunk_size = 1024
nodes = Settings.node_parser.get_nodes_from_documents(documents)

storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)

summary_index = SummaryIndex(nodes, storage_context=storage_context)
vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
keyword_index = SimpleKeywordTableIndex(nodes, storage_context=storage_context)


list_query_engine = summary_index.as_query_engine(
    response_mode = "tree_summarize",
    use_async = True
)
vector_query_engine = vector_index.as_query_engine()
keyword_query_engine = keyword_index.as_query_engine()

list_tool = QueryEngineTool.from_defaults(
query_engine = list_query_engine,
description=(
    "useful for summarization questions about the provided document. retrieves all the context"
)
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine= vector_query_engine,
    description= (
        "useful for retrieving specific context from the provided document"
    )
)

keyword_tool = QueryEngineTool.from_defaults(
    query_engine = keyword_query_engine,
    description = (
        "useful for retrieving specific context from the provided document based on (specific entities within query)"
    )
)


@cl.on_chat_start
async def factory():

    files = None
    while files == None:
                files = await cl.AskFileMessage(
                        content="please upload a PDF file if you want to chat with a document.", 
                        accept=["application/pdf"], max_files=50,
                        max_size_mb=20, timeout=180
                ).send()

    temp_dir = tempfile.mkdtemp()

    for file in files:
          file_path = os.path.join(temp_dir, file.name)

          with open(file_path, "wb") as f_out:
                f_out.write(file.content)

    document_dir += temp_dir

 

    query_engine = RouterQueryEngine(
    selector = LLMSingleSelector.from_defaults(),
    query_engine_tools = [
        list_tool,
        vector_tool,
        keyword_tool
    ],
    streaming = True

) 
    cl.user_session.set("query_engine", query_engine)


@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine")
    response = await cl.make_async(query_engine.query)(message.content)

    response_message = cl.Message(content="")

    for token in response.response_gen:
        await response_message.stream_token(token=token)

    if response.response_txt:
        response_message.content = response.response_txt

    await response_message.send()

