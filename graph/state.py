from typing_extensions import TypedDict
import operator
from typing import Annotated,List,Union
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, ConfigDict
from langgraph.graph import MessagesState
import textwrap

class WebSearchState(TypedDict):
    user_query : str
    search_string : str
    contents : Annotated[list[str], operator.add]

class ContentState(TypedDict):
    user_query: str
    content: str

class OutlineState(TypedDict):
    user_query: str
    outlines: Annotated[list[str], operator.add]
    contents : Annotated[list[str], operator.add]
    consolidated_outline : str

class Analyst(BaseModel):
    affiliation: str = Field(
        description="Primary affiliation of the analyst.",
    )
    name: str = Field(
        description="Name of the analyst."
    )
    role: str = Field(
        description="Role of the analyst in the context of the topic.",
    )
    description: str = Field(
        description="Description of the analyst focus, concerns, and motives.",
    )
    themes_allocated: list[str] = Field(
        description = "A list of themes related to main topic that the analyst is interested in."
    )
    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nThemesAllocated: {self.themes_allocated}\nDescription: {textwrap.fill(self.description, width=80, break_long_words=False, replace_whitespace=False)}\n"


class GenerateAnalystsState(MessagesState):
    user_query: str # Research topic
    outline: str
    max_analysts: int # Number of analysts
    feedback: str # Human feedback
    analysts: List[Analyst] # Analyst asking questions


class InterviewState(MessagesState):
    topic: str
    max_num_turns: int # Number turns of conversation
    context: Annotated[list, operator.add] # Source docs
    analyst: Analyst # Analyst asking questions
    interview: str # Interview transcript
    outline: str
    section: str # Final key we duplicate in outer state for Send() API

class ResearchGraphState(TypedDict):
    topic: str # Research topic
    outline: str #Initial Outline for report
    analysts: List[Analyst] # Analyst asking questions
    sections: Annotated[list, operator.add] # Send() API key
    final_report: str # Final report


