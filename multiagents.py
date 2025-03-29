import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from langchain_groq import ChatGroq
load_dotenv()
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ['GROQ_API_KEY']
search_tools = SerperDevTool() 
llm = ChatGroq(
    model="groq/gemma2-9b-it",
    groq_api_key=GROQ_API_KEY)
#Model Context Protocol

  #backstory = entire prompt about what you want
#   allow_delegation = IF we need to connect with multiple agents
researcher =  Agent(
    role="Senior Researcher",
    goal="Uncover groundbreaking technologies in {topic}",
    backstory="""
        Driven by curiosity, you're at the forefront of innovation,
        eager to explore and share knowledge that could change the world
    """,
    verbose=True,
    memory=True,
    allow_delegation=True,
    tools=[search_tools],
    llm=llm,
)

writer =  Agent(
    role="Writer",
    goal="Narate compelling tech stories about {topic}",
    backstory="""
        While  a flair for simplifying complex topics, you 
        craft engaging narratives that captivate and educate, bringing
        new discoveries to light in an accessible manner.
    """,
    verbose=True,
    memory=True,
    allow_delegation=False,
    tools=[search_tools],
    llm=llm,
)

research_task = Task(
    description=(
        "Identify the next big trend in  {topic}."
        "Focus on identifying pros and cons and the overall narrative."
        "Your final report should clearly articulate the key points,"
        "its market opportunities, and potential risks."
    ),
    expected_output="A comprehensive 3 paragraphs long report on the latest AI trend.",
    tools=[search_tools],
    agent=researcher
)

writer_task = Task(
    description=("Compose an insightful article on {topic}."
    "Focus on the latest trends and how it's impacting the industry."
    "This article should be easy to understand, engaging, and positive."
    ),
    expected_output="A 4 paragraph article on {topic} advancements formatted as markdown",
    tools=[search_tools],
    agent=writer,
    async_execution=False, 
    output_file="new-blog-post.md"
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writer_task],
    process=Process.sequential
)

topic = input("Please enter a topic for blog post: ")
result = crew.kickoff(
    inputs={
        "topic": topic
    }
)

print(result)
