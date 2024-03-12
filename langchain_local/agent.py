import os

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
from dotenv import load_dotenv, find_dotenv


from tools.rag_tool import rag_one, rag_two
from tools.serper_tool import serper_tool

_ = load_dotenv(find_dotenv())

api_key = os.getenv('OPENAI_API_KEY')

tools = [rag_one, rag_two, serper_tool]

functions = [convert_to_openai_function(f) for f in tools]
model = ChatOpenAI(temperature= 0).bind_functions(functions=functions)
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


