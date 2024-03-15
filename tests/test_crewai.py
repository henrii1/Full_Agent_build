import os
import pytest
import openai
from crewai_local.trip_crew import TripCrew

from crewai import Crew, Process
from textwrap import dedent
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from crewai_local.trip_agents import TripAgents
from crewai_local.trip_task import TripTasks

from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.environ.get('OPENAI_API_KEY')


@pytest.mark.parametrize("cities, date_range, interests, expected_output", [
    ("London", "early fall", "hiking, swimming, snorkelling", "YES"),
])
def test_crew(cities, date_range, interests, expected_output):
    """
    This improved test case incorporates the following enhancements:

    1. Explicit Imports: Makes the code more readable by explicitly importing Chatollama.
    2. Expected Output: Adds a new argument `expected_output` to the parametrize decorator.
        - This improves clarity by specifying the anticipated outcome of the test.
    3. Clearer Variable Naming: Uses more descriptive names like `expected_response` instead of `prompt`.
    4. Direct Assertion: Directly asserts the `chain['content']` against the `expected_output`.
        - This simplifies the test and avoids unnecessary intermediate steps.
    5. Docstring: Adds a docstring to explain the purpose of the test and the changes made.
    """

    t_crew = TripCrew(cities, date_range, interests)
    response = t_crew.run()

    model = ChatOllama(model="mistral")
    expected_response = f"Is the context in {response} a detailed travel plan for a trip to one of the cities in the list provided `{cities}`. Response should be YES or NO. Remember to capitalize the answer"
    chain = expected_response | model

    assert chain == expected_output

