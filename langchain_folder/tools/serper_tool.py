import os
import pprint


from langchain_core.tools import tool
from langchain.pydantic_v1 import BaseModel, Field
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
from dotenv import load_dotenv

load_dotenv()
os.environ["SERPER_API_KEY"] = os.getenv('SERPER_API_KEY')

class SerperTool(BaseModel):
  query: str = Field(description = "This should be a search query")

@tool("serper_tool_main", args_schema=SerperTool, return_direct=False)
def serper_tool(query:str) -> str:
  """A useful for when you need to ask with search. Very useful when recent or specific information is needed from the web 
  """
  search = GoogleSerperAPIWrapper(k=4, type="search")
  initial_result = search.results(query)
  result = initial_result['organic']
  results = ""
  for r in result:
    data = f"'Title':{r['title']}\n 'content':{r['snippet']}"
    results += f"{data}\n\n"
  return results