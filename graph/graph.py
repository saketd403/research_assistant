from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field, ConfigDict
from langgraph.types import StreamWriter
from langgraph.checkpoint.memory import MemorySaver
import os

from graph.state import WebSearchState, OutlineState, ContentState, ResearchGraphState
from graph.nodes import (LLM, tavily_search, query_transform, create_outline, 
continue_to_outlines, websearch_subgraph, consolidate_outlines,
GenerateAnalystsState, generate_analysts, feedback_analyst, continue_feedback_analyst,
route_interview, InterviewState, generate_question_interview, search_web_interview, 
answer_question_interview, save_interview, write_section, conduct_interview, initiate_interviews,
write_report)


def build_websearch_graph():

    websearch_builder = StateGraph(state_schema=WebSearchState)

    websearch_builder.add_node("query_transform",query_transform)
    websearch_builder.add_node("tavily_search",tavily_search)

    websearch_builder.add_edge(START,"query_transform")
    websearch_builder.add_edge("query_transform","tavily_search")
    websearch_builder.add_edge("tavily_search",END)

    memory = MemorySaver()
    graph = websearch_builder.compile(checkpointer=memory)

    return graph

def build_outline_graph():

    outline_builder = StateGraph(state_schema=OutlineState)

    outline_builder.add_node("websearch_topic",websearch_subgraph)
    outline_builder.add_node("create_outline",create_outline)
    outline_builder.add_node("consolidate_outlines",consolidate_outlines)

    outline_builder.add_edge(START,"websearch_topic")
    outline_builder.add_conditional_edges("websearch_topic",continue_to_outlines)
    outline_builder.add_edge("create_outline","consolidate_outlines")
    outline_builder.add_edge("consolidate_outlines",END)

    memory = MemorySaver()
    graph = outline_builder.compile(checkpointer=memory)

    return graph

def build_analyst_graph():

    analyst_builder = StateGraph(state_schema=GenerateAnalystsState)

    analyst_builder.add_node("generate_analysts",generate_analysts)
    analyst_builder.add_node("feedback_analyst",feedback_analyst)

    analyst_builder.add_edge(START,"generate_analysts")
    analyst_builder.add_edge("generate_analysts","feedback_analyst")
    analyst_builder.add_conditional_edges("feedback_analyst",continue_feedback_analyst)

    memory = MemorySaver()
    graph = analyst_builder.compile(checkpointer=memory)

    return graph

def build_interview_graph():

    interview_builder = StateGraph(state_schema=InterviewState)

    interview_builder.add_node("generate_question_interview",generate_question_interview)
    interview_builder.add_node("search_web_interview",search_web_interview)
    interview_builder.add_node("answer_question_interview",answer_question_interview)
    interview_builder.add_node("save_interview",save_interview)
    interview_builder.add_node("write_section",write_section)

    interview_builder.add_edge(START,"generate_question_interview")
    interview_builder.add_edge("generate_question_interview","search_web_interview")
    interview_builder.add_edge("search_web_interview","answer_question_interview")
    interview_builder.add_conditional_edges("answer_question_interview",route_interview)
    interview_builder.add_edge("save_interview","write_section")
    interview_builder.add_edge("write_section",END)

    memory = MemorySaver()
    graph = interview_builder.compile(checkpointer=memory)

    return graph

def build_final_report():

    report_builder = StateGraph(state_schema=ResearchGraphState)

    report_builder.add_node("conduct_interview",conduct_interview)
    report_builder.add_node("write_report",write_report)

    report_builder.add_conditional_edges(START,initiate_interviews)
    report_builder.add_edge("conduct_interview","write_report")
    report_builder.add_edge("write_report",END)

    memory = MemorySaver()
    graph = report_builder.compile(checkpointer=memory)

    return graph



