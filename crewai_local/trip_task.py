from crewai import Task
from textwrap import dedent
from datetime import date


class TripTasks():

    def __tip_selection(self):
        return "If you do your BEST WORK, I'll tip you $100!"

    def identify_task(self, agent, origin, cities, interests, range):
        return Task(description=dedent(f"""
                               Analyze and select the best city for the trip based on specific criteria
                               such as weather patterns, seasonal events and travel costs. This task
                               involves comparing multiple cities, considering factors like current weather
                               conditions, upcoming cultural or seasonal events and overall travel expenses.text=
                               Your final answer must be a detailed report on the chosed city
                               and everything you found out about it, including the actual flight costs, weather forecase and attractions.
                               {self.__tip_selection()}
                               
                               Traveling from: {origin}
                               City Options: {cities}
                               Trip Date: {range}
                               Traveler Interests: {interests}
                               """
                               ),
            expected_output="identified tasks to be performed",
            agent=agent,
            async_execution=True,

        )
    
    def gather_task(self, agent, context, origin, interests, range):
        return Task(description=dedent(f"""
                                       As a local expert on this city you must compile an in-depth guide for
                                       someone traveling there and wanting to have THE BEST trip ever!
                                       
                                       Gather information about key attraction, local customs, special events and
                                       daily activitiy recommendations.
                                       Find the best spots to go, the kind of places only a local would know.
                                       This guide should provide a thorough overview of what the city has to offer
                                       including hidden gems, cultural hotspots, must-visit landmarks, weather forecasts,
                                       and high level costs.text=
                                       The final answer must be a comprehensive city guide,
                                       rich in cultural insights and practical tips,
                                       tailored to enhance the travel experience.
                                       {self.__tip_selection()}
                                       
                                       Trip Date: {range}
                                       Traveling from: {origin}
                                       Traveler Interests: {interests}
                                       """),
                                        expected_output='list of possible tasks for a given location grouped together',
                                       agent=agent,
                                       context=context,

        )
    
    def plan_task(self, agent, origin, interests, range, context):
        return Task(
            description=dedent(f"""
                               Expand this guide into a full 7-day travel itinerary with detailed
                               per-day plans including weather forecasts, places to eat, packing suggestions
                               and a budget breakdown.text=
                               You MUST suggest actual places to visit, actual hotels to stay
                               and actual restaurants to go to.
                               
                               This itinerary should cover all aspects of the trip,
                               from arrival to departure, integrating the city guide information with
                               practical travel logistics.text=
                               Your final answer MUST be a complete expanded travel plan, formatted as
                               markdown, encompassing a daily schedule, anticipated weather conditions, recommended clothings
                               items to pack, and a detailed budget ensuring THE BEST TRIP EVER, Be specific
                               and give reasons why you picked a specific place, what makes the place special.
                               
                               {self.__tip_selection()}
                               
                               Trip Date: {range}
                               Traveling from: {origin}
                               Traveler Interests: {interests}
                               """),
                               agent = agent,
                               expected_output="A detailed plan of the activities to be carried out",
                               context = context
        )

