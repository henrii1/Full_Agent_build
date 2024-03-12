import os
import pprint

from langchain_core.tools import Tool
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

serper_api_key = os.getenv('SERPER_API_KEY')

search = GoogleSerperAPIWrapper(serper_api_key = serper_api_key)

serper_tool = Tool(
    name = "Intermediate Answer",
    func = search.run,
    description="useful for when you need to ask with search"
)