import os

from crewai import Crew, Process
from textwrap import dedent
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from trip_agents import TripAgents
from trip_task import TripTasks

from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

class TripCrew:

    def __init__(self, origin, cities, date_range, interests):
        
        self.origin = origin
        self.cities = cities
        self.date_range = date_range
        self.interests = interests
        self.mistral = ChatOllama(model="mistral")
        self.openai = ChatOpenAI(api_key=api_key, temperature=0)

    def run(self):
        agents = TripAgents()
        tasks = TripTasks()

        city_selector_agent = agents.city_selection_agent()
        local_expert_agent = agents.local_expert()
        travel_concierge_agent = agents.travel_concierge()

        identify_task = tasks.identify_task(
            agent=city_selector_agent,
            origin=self.origin,
            cities=self.cities,
            interests=self.interests,
            range=self.date_range
        )

        gather_task = tasks.gather_task(
            agent=local_expert_agent,
            origin=self.origin,
            interests=self.interests,
            range=self.date_range,
            context=[identify_task]

        )

        plan_task = tasks.plan_task(
            agent=travel_concierge_agent,
            origin=self.origin,
            interests=self.interests,
            range=self.date_range,
            context=[gather_task]
        )

        crew = Crew(
            agents=[city_selector_agent, local_expert_agent, travel_concierge_agent],
            tasks=[identify_task, gather_task, plan_task],
            process= Process.hierarchical,
            manager_llm=self.openai
        )

        result = crew.kickoff()
        return result
    

if __name__ == "__main__":
    import streamlit as st

    st.title("Trip Planner Crew!!")
    st.write("""Input a location, a list of cities you have in mind, your proposed
                travel date and you interests. Watch this agent plan a costed trip for you""")

    location = st.text_area("Enter you location")
    cities = st.text_area("Enter the lis of cities you intend to visit")
    date_range = st.text_area("Enter the proposed data range for the trip")
    interests = st.text_area("Enter a list of things you're interested in or hope to do during the trip")

    if st.sidebar.button("Build a plan"):
        with st.spinner(text = "Generating"):
            trip_crew = TripCrew(
                origin = location,
                cities = cities,
                date_range = date_range,
                interests = interests
            )
            result = trip_crew.run()
            st.markdown(result)






        