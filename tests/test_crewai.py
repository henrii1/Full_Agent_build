import os
import pytest
from crewai_local.trip_crew import TripCrew

from crewai import Crew, Process
from textwrap import dedent
from langchain_community.chat_models import ChatOllama
from trip_agents import TripAgents
from trip_task import TripTasks

from dotenv import load_dotenv
load_dotenv()



@pytest.mark.parametrize("cities, date_range, interests", [("London", "early fall", "hiking, swiming, snorkelling")])
def test_crew(cities, date_range, interests):
    t_crew = TripCrew(cities, date_range, interests)
    response = t_crew.run()
    model = Chatollama(model = "mistral")
    prompt = f"Is the context in {response} a detailed travel plan
                for a trip to one of the cities in the list provided
                after in backticks `{cities}`. Response should be YES or NO
                remember to capitalize the answer"
    chain = prompt | model
    assert chain['content'] == "YES"

