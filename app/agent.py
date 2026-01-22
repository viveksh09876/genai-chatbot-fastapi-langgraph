import os
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY=os.environ.get("TAVILY_API_KEY")
OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage

def get_response_from_ai_agent(
    llm_id: str,
    query: list[str],
    allow_search: bool,
    system_prompt: str,
    provider: str
) -> str: 
    """
    Returns response from AI agent based on:
    - provider (Groq/OpenAI)
    - model name
    - optional Tavily search tool
    """

    if provider == "Groq":
        llm  = ChatGroq(model=llm_id)
    elif provider == "OpenAI":
        llm = ChatOpenAI(model=llm_id)
    else:
        return "Invalid provider selected"

    tools = [TavilySearchResults(max_results=2)] if allow_search else []

    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=system_prompt
    )

    state = { "message": query }
    response = agent.invoke(state)

    messages = response.get("messages", [])

    ai_messages = [
        message.content for message in messages if isinstance(message, AIMessage)
    ]

    return ai_messages[-1] if ai_messages else "No response received"

